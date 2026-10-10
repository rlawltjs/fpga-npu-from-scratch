# LeNet-5 NPU (Stage 1 of LeNet → MobileNet → YOLO11n-mini)

순수 Verilog(RTL)로 작성한 LeNet-5 추론 가속기입니다. Zynq-7020급 소형 FPGA를
목표로, "단순하지만 완전히 동작하는" 시퀀셜 NPU를 먼저 확립하고, 이후 단계
(MobileNet의 depthwise conv, YOLO11n-mini의 CSP 블록 등)에서 파이프라이닝/
병렬화로 확장하기 위한 기반(control FSM, 메모리 맵, 양자화 방식)을 마련하는
것이 목표입니다.

Icarus Verilog로 시뮬레이션까지 완료해 전체 파이프라인(9단계)이 정상적으로
끝까지 실행되는 것을 확인했습니다.

## 디렉토리 구조

```
lenet_npu/
├── rtl/
│   ├── lenet_top.v         # 최상위: 메모리 인스턴스, 엔진 재사용, 레이어 시퀀서 FSM
│   ├── conv2d_engine.v      # 런타임 설정 가능한 conv 엔진 (C1, C3 재사용)
│   ├── maxpool2x2.v         # 런타임 설정 가능한 2x2 maxpool 엔진 (S2, S4 재사용)
│   ├── fc_engine.v          # 런타임 설정 가능한 FC(행렬-벡터곱) 엔진 (F5,F6,OUT 재사용)
│   ├── argmax_unit.v        # 최종 클래스 인덱스 산출
│   ├── bram_dp.v            # 범용 동기식 메모리 (feature map / weight / bias)
│   └── requant_func.vh      # bias 반영 + shift + saturate + ReLU 공용 함수
├── tb/
│   └── tb_lenet_top.v       # 전체 파이프라인 end-to-end 테스트벤치
├── scripts/
│   └── gen_weights.py       # 시뮬레이션용 더미 양자화 weight/bias/image hex 생성
└── sim/                     # gen_weights.py 실행 결과 (*.hex) + 시뮬레이션 산출물
```

## 아키텍처 개요

```
img(1,32,32) --C1 conv 1→6ch--> (6,28,28) --S2 pool--> (6,14,14)
             --C3 conv 6→16ch-> (16,10,10) --S4 pool--> (16,5,5) [=400 flat]
             --F5 fc 400→120--> (120) --F6 fc 120→84--> (84)
             --OUT fc 84→10---> (10) --ARGMAX--> class index
```

**핵심 설계 포인트: 엔진 재사용.** conv2d_engine, maxpool2x2, fc_engine은 각각
**딱 하나의 인스턴스**만 존재하고, in_ch/out_ch/in_size (또는 in_dim/out_dim)를
**런타임 입력 포트**로 받습니다. 최상위 FSM이 레이어가 바뀔 때마다 이 설정값과
메모리 연결(mux)을 갈아끼워서 같은 하드웨어를 C1/C3에, 같은 풀링 하드웨어를
S2/S4에, 같은 FC 하드웨어를 F5/F6/OUT에 반복 사용합니다. 실리콘 재사용률을
높이는 실제 NPU 설계의 기본 아이디어이기도 합니다.

**메모리**: 이미지용 `img_mem` 1개, 레이어 간 activation을 주고받는 핑퐁 버퍼
`fm_a`/`fm_b` 2개, 그리고 레이어별 weight/bias ROM 10개(`w_conv1/2`,
`b_conv1/2`, `w_fc1/2/3`, `b_fc1/2/3`)로 구성됩니다. `fm_a`↔`fm_b`가 매 레이어
소스/목적지 역할을 교대하므로 별도 DMA 없이 추론이 완결됩니다.

**양자화**: 8비트 signed 고정소수점 activation/weight, 32비트 signed
accumulator. 각 레이어 출력 시 `acc >>> SHIFT` 로 재양자화 후 saturate,
필요 시 ReLU 적용 (`requant_func.vh`). FC의 마지막 출력층(OUT)만
`apply_relu=0`으로 raw score를 유지해 argmax에 사용합니다.

**제어 방식 (Stage 1 설계 의도)**: 각 MAC 연산은 "주소 구동 1사이클 + 데이터
캡처/누산 1사이클"의 2-cycle 순차 방식입니다. 파이프라이닝이나 병렬 MAC 배열이
아니라, 정확성과 검증 용이성을 최우선한 baseline입니다. 이후 MobileNet 단계에서
depthwise/pointwise 분리에 맞춰 파이프라인을 추가하고, YOLO 단계에서 병렬
MAC 배열로 확장하는 것을 권장합니다.

## 실행 방법 (Icarus Verilog)

```bash
# 1) 시뮬레이션용 더미 weight/bias/image hex 생성
cd lenet_npu
python3 scripts/gen_weights.py

# 2) 컴파일
iverilog -g2005 -I rtl -o sim/lenet_tb \
    rtl/bram_dp.v rtl/conv2d_engine.v rtl/maxpool2x2.v \
    rtl/fc_engine.v rtl/argmax_unit.v rtl/lenet_top.v \
    tb/tb_lenet_top.v

# 3) 실행
vvp sim/lenet_tb
```

**실제 확인된 실행 결과 (더미 가중치 기준):**
```
=====================================================
 LeNet-5 NPU inference complete
 Predicted class index : 0
 Total clock cycles    : 868614
=====================================================
```

100MHz 클럭 기준 868,614 사이클 ≈ 8.69ms/frame (Stage 1 순차 방식 기준치;
파이프라이닝 적용 시 크게 단축 가능). 파형은 `sim/waves.vcd`에 저장되며
`gtkwave sim/waves.vcd`로 확인할 수 있습니다.

## ⚠️ 중요: 더미 가중치입니다

`scripts/gen_weights.py`가 생성하는 weight/bias/image는 **구조적 검증용
난수**이며, README에서 언급된 실제 학습된 PyTorch 모델의 값이 아닙니다.
이 테스트벤치는 "데이터패스·제어 FSM·메모리 주소 매핑이 끝까지 올바르게
동작하는가"를 검증하는 것이 목적이지, "정확도가 높은 분류기"를 만드는
것이 목적이 아닙니다.

실제 정확도를 검증하려면:
1. PyTorch로 LeNet-5를 학습 (또는 MNIST 등으로 대체 학습)
2. 각 weight/bias/activation을 이 설계와 동일한 INT8 / INT32-bias 고정소수점
   규약으로 양자화 (post-training quantization)
3. `gen_weights.py`의 `write_hex_i8` / `write_hex_i32` 형식 그대로 hex 파일을
   생성해 `sim/` 아래 동일한 파일명으로 교체

## 다음 단계 (MobileNet, YOLO11n-mini)

이 Stage 1에서 확립한 것 중 그대로 재사용되는 것:
- `bram_dp.v` 메모리 모듈과 핑퐁 버퍼 구조
- `requant_func.vh` 양자화/ReLU 공용 함수
- "엔진을 런타임 파라미터로 재사용" 설계 철학
- 레이어 시퀀서 FSM 패턴 (start pulse → done 대기 → 다음 레이어)

MobileNet 단계에서 추가로 필요한 것:
- Depthwise separable conv (채널 간 reduction 없는 별도 dataflow)
- BatchNorm(가중치에 fold 처리), ReLU6
- On-chip에 다 못 올라가는 activation을 위한 tiling

YOLO11n-mini 단계에서 추가로 필요한 것:
- SiLU 활성화 (LUT 근사)
- CSP 블록 (split-concat), SPPF (멀티 maxpool+concat)
- FPN/PAN 넥 (upsample+concat, 멀티스케일 버퍼 관리)
- NMS는 PL이 아닌 PS(ARM)에서 처리하는 하이브리드 구조 권장
