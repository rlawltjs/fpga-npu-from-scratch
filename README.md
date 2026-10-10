# LeNet-5 NPU (Stage 1 of LeNet → MobileNet → YOLO11n-mini)

순수 Verilog(RTL)로 작성한 LeNet-5 추론 가속기입니다. Zynq-7020급 소형 FPGA를
목표로, "단순하지만 완전히 동작하는" 시퀀셜 NPU를 먼저 확립하고, 이후 단계
(MobileNet의 depthwise conv, YOLO11n-mini의 CSP 블록 등)에서 파이프라이닝/
병렬화로 확장하기 위한 기반(control FSM, 메모리 맵, 양자화 방식)을 마련하는
것이 목표입니다.

Icarus Verilog로 시뮬레이션까지 완료해 전체 파이프라인(9단계)이 정상적으로
끝까지 실행되는 것을 확인했고, 나아가 Python 골든 모델과의 레이어별
bit-exact 비교로 **연산 결과 자체가 의도한 대로 정확한지**까지 검증했습니다
(그 과정에서 실제 타이밍 버그 1건을 찾아 수정 — 아래 "검증 방법론" 참고).

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
│   ├── tb_lenet_top.v       # 전체 파이프라인 end-to-end 테스트벤치
│   └── tb_dump_layers.v     # 레이어별 중간 결과를 hex로 덤프 (골든 모델 비교용)
├── scripts/
│   ├── gen_weights.py       # 시뮬레이션용 더미 양자화 weight/bias/image hex 생성
│   ├── golden_model.py      # RTL과 bit-exact 동일한 고정소수점 연산의 NumPy 재구현
│   └── compare_layers.py    # 골든 모델 vs RTL 덤프 레이어별 자동 비교 (PASS/FAIL)
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

**제어 방식 (Stage 1 설계 의도)**: 각 MAC 연산은 "주소 구동 1사이클 + 메모리
출력 레지스터 대기 1사이클 + 데이터 캡처/누산 1사이클"의 3-cycle 순차 방식입니다
(`bram_dp`의 2-cycle read latency 때문 — 자세한 내용은 아래 "검증 방법론"의
버그 사례 참고). 파이프라이닝이나 병렬 MAC 배열이 아니라, 정확성과 검증
용이성을 최우선한 baseline입니다. 이후 MobileNet 단계에서 depthwise/pointwise
분리에 맞춰 파이프라인을 추가하고, YOLO 단계에서 병렬 MAC 배열로 확장하는
것을 권장합니다.

## 검증(Verification) 방법론

이 프로젝트를 "제대로" 검증하려면 아래 4단계를 순서대로 밟아야 합니다.
1단계만으로는 "끝까지 실행됐다"만 알 수 있을 뿐, "정확하게 계산했다"는
전혀 보장하지 못합니다. 실제로 이 저장소는 2단계에서 진짜 타이밍 버그를
찾아서 고쳤습니다 (아래 "발견된 버그" 참고).

| 단계 | 검증 대상 | 도구 |
|---|---|---|
| 1. 구조적 검증 | RTL이 hang 없이 끝까지 실행되는가 | `tb/tb_lenet_top.v` |
| 2. 수치적 검증 | RTL 연산이 의도한 고정소수점 수식과 bit-exact로 일치하는가 | `scripts/golden_model.py` + `scripts/compare_layers.py` |
| 3. 정확도 검증 | 실제 데이터로 잘 분류하는가 | 실제 학습된 weight + MNIST 테스트셋 |
| 4. 하드웨어 검증 | 합성/타이밍/실보드 동작이 맞는가 | Vivado 합성 리포트 + 보드 테스트 |

### 1단계: 구조적 검증 (기본 실행)

```bash
cd lenet_npu
python3 scripts/gen_weights.py
iverilog -g2005 -I rtl -o sim/lenet_tb \
    rtl/bram_dp.v rtl/conv2d_engine.v rtl/maxpool2x2.v \
    rtl/fc_engine.v rtl/argmax_unit.v rtl/lenet_top.v \
    tb/tb_lenet_top.v
vvp sim/lenet_tb
```
출력이 hang 없이 `Predicted class index`를 찍으면 통과. 하지만 이 숫자가
**맞는 숫자인지는 이 단계만으로 알 수 없습니다.**

### 2단계: 수치적 검증 (골든 모델 비교) — 가장 중요한 단계

`scripts/golden_model.py`는 RTL과 **완전히 동일한 정수 연산 규칙**
(conv valid, bias add, `acc >>> SHIFT`, saturate, ReLU)을 NumPy로
재구현합니다. `tb/tb_dump_layers.v`는 매 레이어가 끝나는 순간 `fm_a`/`fm_b`
메모리 내용을 hex로 덤프합니다. 둘을 비교하면 RTL 버그를 레이어 단위로
정확히 잡아낼 수 있습니다.

```bash
# 레이어별 중간값을 hex로 덤프하는 테스트벤치 실행
iverilog -g2005 -I rtl -o sim/dump_tb \
    rtl/bram_dp.v rtl/conv2d_engine.v rtl/maxpool2x2.v \
    rtl/fc_engine.v rtl/argmax_unit.v rtl/lenet_top.v \
    tb/tb_dump_layers.v
vvp sim/dump_tb

# 골든 모델과 레이어별 bit-exact 비교 (자동 PASS/FAIL 판정)
python3 scripts/compare_layers.py
```

**정상적인 출력:**
```
[PASS] C1     shape=(6, 28, 28)
[PASS] S2     shape=(6, 14, 14)
[PASS] C3     shape=(16, 10, 10)
[PASS] S4     shape=(16, 5, 5)
[PASS] F5     shape=(120,)
[PASS] F6     shape=(84,)
[PASS] OUT    shape=(10,)

ALL LAYERS MATCH -- RTL is numerically bit-exact with the golden model.
```

#### 🐛 발견된 버그 (실제 사례)

이 검증 과정에서 실제로 버그를 하나 찾았습니다. 최초 버전은 첫 픽셀 출력이
`x`(undefined)였고, 최종 예측 클래스도 골든 모델(클래스 6)과 다른 클래스 0을
냈습니다.

**원인**: `bram_dp`는 동기식 read 메모리라 "주소 안정화 1사이클 + 메모리
출력 레지스터 반영 1사이클", 총 **2사이클의 read latency**가 있습니다.
그런데 conv/pool/fc/argmax 엔진들은 전부 "주소 설정(1사이클) → 데이터
사용(다음 1사이클)"만 기다리는 2-state FSM(`S_ADDR`→`S_READ`)으로
짜여 있어서, 매 항마다 데이터가 한 사이클씩 밀려 읽혔습니다. 특히 각 루프의
첫 항은 아예 정의되지 않은 이전 상태를 곱해버렸죠.

**수정**: 모든 엔진에 `S_WAIT` 상태를 추가해 주소→데이터 latency를
정확히 2사이클로 맞췄습니다 (`S_ADDR`→`S_WAIT`→`S_READ`). 이 커밋 전후로
`compare_layers.py`를 돌려보면 차이가 확실히 드러납니다.

이 사례가 보여주는 것: **시뮬레이션이 "끝까지 돈다"는 것과 "정확하다"는
것은 완전히 다른 문제**입니다. 특히 메모리 read latency를 다루는 순차
가속기 설계에서는 골든 모델과의 bit-exact 비교가 사실상 필수입니다.

**버그 수정 후 (현재 코드) 실제 실행 결과:**
```
=====================================================
 LeNet-5 NPU inference complete
 Predicted class index : 6
 Total clock cycles    : 1297966
=====================================================
```
골든 모델의 예측(클래스 6, OUT scores `[49,121,-76,91,123,18,127,-54,-21,-121]`,
최댓값 127이 인덱스 6)과 정확히 일치합니다. `S_WAIT` 상태 추가로 사이클 수는
868,614 → 1,297,966으로 늘었지만 (~1.5배), 이는 정확성을 위한 정당한 비용입니다.
파형은 `tb/tb_lenet_top.v`의 `$dumpfile` 경로에 `sim/waves.vcd`로 저장되며
`gtkwave sim/waves.vcd`로 확인할 수 있습니다 (용량이 크므로 필요할 때만 생성 권장).

### 3단계: 정확도 검증 (실제 학습된 가중치 필요)

지금까지 통과한 건 "RTL이 의도한 수식을 정확히 계산한다"까지입니다. 이 수식
자체가 "숫자를 잘 분류하는 좋은 모델"인지는 완전히 별개 문제 — 지금 hex
파일들은 랜덤 값이라 애초에 아무 의미도 없습니다. 정확도를 검증하려면:

1. **PyTorch로 LeNet-5 학습** (MNIST 등으로): `nn.Conv2d(1,6,5) → ReLU →
   MaxPool2d(2) → Conv2d(6,16,5) → ReLU → MaxPool2d(2) → Flatten →
   Linear(400,120) → ReLU → Linear(120,84) → ReLU → Linear(84,10)` 구조로
   레이어 이름/차원을 이 RTL과 정확히 맞춰서 학습
2. **Post-training quantization (PTQ)**: 각 레이어의 activation 분포를 보고
   INT8 스케일(`SHIFT` 값)을 레이어마다 산정 — 지금 RTL은 전 레이어 공통
   `SHIFT` 하나만 쓰므로, 처음엔 단순하게 모든 레이어에 같은 SHIFT를 적용해보고
   정확도가 너무 떨어지면 레이어별 SHIFT를 RTL에 런타임 파라미터로 추가
   (지금 구조상 `conv2d_engine`/`fc_engine`에 `shift_amt` 입력 포트만
   추가하면 되므로 확장이 어렵지 않습니다)
3. `scripts/gen_weights.py`의 `write_hex_i8`/`write_hex_i32` 형식 그대로,
   학습된 뒤 양자화한 실제 weight/bias로 `sim/*.hex`를 교체
4. **테스트셋 전체를 돌려서 정확도 비교**: PyTorch(float) 모델의 top-1
   정확도와, 같은 이미지들을 `golden_model.py`(정수 고정소수점)로 돌린
   정확도를 비교 — 이 차이가 "양자화로 인한 손실"입니다. 이후 RTL
   시뮬레이션도 소수의 이미지에 대해 golden_model과 대조해 계속 bit-exact를
   유지하는지 확인
5. 이미지 1장당 시뮬레이션에 실제 걸리는 시간(수 초~수십 초, 위 실측 참고)을
   감안하면 전체 테스트셋(수천 장)을 RTL 시뮬레이션으로 도는 건 비현실적이니,
   **golden_model.py로 정확도를 측정하고, RTL은 대표 샘플 수십 장만 bit-exact
   비교**하는 것이 현실적인 전략입니다 (golden_model이 RTL의 정확한 대리
   모델임을 2단계에서 이미 증명했으므로 이 전략이 타당합니다)

### 4단계: 하드웨어 검증 (합성 · 타이밍 · 실보드)

시뮬레이션 통과와 "실제 FPGA에서 동작"은 또 다른 문제입니다. 특히 아래
항목들은 시뮬레이션에서는 절대 안 드러나고 합성/구현 단계에서만 드러납니다.

- **Synthesis-simulation mismatch 점검**: `$readmemh`로 초기화한 ROM이
  Vivado에서 실제로 BRAM 초기값으로 합성되는지 확인 (합성 후 netlist
  시뮬레이션으로 재검증 — 순수 RTL 시뮬레이션과 다른 결과가 나오면 보통
  latch 추론이나 초기화 문제)
- **자원 사용량 확인**: `w_fc1` 하나가 48,000 × 8bit ≈ 375Kbit인데, 이런
  weight ROM들을 BRAM으로 합성했을 때 Zynq-7020의 총 BRAM(4.9Mbit)을
  넘지 않는지 Vivado 합성 리포트로 확인 — 넘으면 weight를 DDR에 두고
  스트리밍하는 구조로 바꿔야 함
- **타이밍 클로저**: 100MHz(또는 목표 클럭)에서 setup/hold 위반이 없는지
  구현(implementation) 후 타이밍 리포트 확인. 이 설계는 조합 로직이 짧은
  편(주소 계산 + MAC 1개)이라 큰 문제는 없을 가능성이 높지만, `requant_relu`
  함수의 shift+saturate 로직이 큰 SHIFT/ACC_W에서 길어지면 확인 필요
- **AXI 인터페이스 추가 후 검증**: 지금은 이미지/가중치를 `$readmemh`로
  시뮬레이션 시점에 미리 넣지만, 실보드에서는 PS(ARM)가 AXI-Lite/Stream +
  DMA로 이미지를 넣어줘야 합니다. 이 인터페이스를 추가한 뒤 다시 2단계
  (golden model 비교)를 반복해서 통합 이후에도 bit-exact가 유지되는지 확인
- **실보드 검증**: UART로 `class_idx`를 출력하거나 LED에 표시해서, 알고 있는
  정답 이미지 몇 장을 넣었을 때 실제로 기대한 클래스가 나오는지 확인. 이때
  PC에서 시뮬레이션했던 것과 같은 이미지/가중치를 써야 1:1 비교가 됩니다

## ⚠️ 중요: 더미 가중치입니다

`scripts/gen_weights.py`가 생성하는 weight/bias/image는 **구조적/수치적
검증용 난수**입니다 (1~2단계 통과 확인용). 실제 학습된 PyTorch 모델의 값이
아니므로 "분류를 잘 한다"는 의미가 전혀 없습니다. 3단계를 참고해 실제
가중치로 교체하세요.

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
