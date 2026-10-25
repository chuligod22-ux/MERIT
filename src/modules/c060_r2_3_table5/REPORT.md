# 1. C060 R2-3 최종 Table 5 결과 보고서

## 1.1 판정

**PASS — R1-2 ALIGNED / R2-3 RESPONSE-READY / MANUSCRIPT APPLIED**

초기 ON/OFF 진단표에 C015의 과거 절대 PSNR·LPIPS를 붙이지 않고, R1-2에서 최종 채택한 C010·C039·C049 고정 경로의 paired 변화량으로 Table 5를 재구성했다. NAFNet에서 빠져 있던 LPIPS는 C010의 동일 동결 은행·체크포인트·route를 재실행해 추가했다.

## 1.2 평가 계약

| Backbone | 확인은행 | 단위 | 모델 seed | 조건 |
|---|---|---:|---:|---:|
| NAFNet | C010 unused-condition | 68 parent·98 crop | 3 | low/mid/high |
| MPRNet | C036 ISO200 | 133 parent·191 crop | 3 | low/mid/high |
| Restormer | C040 ISO400 | 302 parent·450 crop | 3 | low/mid/high |

Estimand는 같은 입력과 source reference에서 `최종 고정 MERIT route − 대응 backbone baseline`이다. Crop·열화 반복·모델 seed를 physical parent 안에서 평균한 뒤 parent를 5,000회 bootstrap했다.

## 1.3 최종 9행 결과

| Backbone | 조건 | ΔPSNR dB [95% CI] | ΔLPIPS-Alex [95% CI] |
|---|---|---:|---:|
| NAFNet | low | +0.666654 [+0.665202,+0.668093] | -0.049406 [-0.050968,-0.047840] |
| NAFNet | mid | +0.003618 [+0.003472,+0.003764] | +0.002018 [+0.001974,+0.002061] |
| NAFNet | high | +0.082088 [+0.080587,+0.083632] | +0.023515 [+0.022765,+0.024259] |
| MPRNet | low | +0.036667 [+0.035168,+0.038309] | -0.003389 [-0.003464,-0.003319] |
| MPRNet | mid | +0.010160 [+0.008144,+0.012424] | -0.001294 [-0.001322,-0.001265] |
| MPRNet | high | +0.006107 [+0.005535,+0.006671] | -0.001901 [-0.001955,-0.001844] |
| Restormer | low | +0.532687 [+0.527466,+0.538283] | -0.032808 [-0.033506,-0.032135] |
| Restormer | mid | +0.040119 [+0.036659,+0.043851] | -0.007946 [-0.008476,-0.007423] |
| Restormer | high | +0.014968 [+0.014105,+0.015878] | -0.000727 [-0.001011,-0.000425] |

NAFNet의 세 조건 pooled 결과는 ΔPSNR `+0.250787 dB` [95% CI `+0.250127,+0.251435`]와 ΔLPIPS `-0.007958` [`-0.008230,-0.007681`]이다.

## 1.4 해석

1. PSNR은 9/9 조건에서 평균과 CI 하한이 모두 양수다. 따라서 R1-2의 세 평가 backbone PSNR 비감소 결론과 일치한다.
2. MPRNet과 Restormer는 6/6 조건에서 LPIPS도 개선됐다.
3. NAFNet LPIPS는 low와 pooled에서는 개선됐지만 mid·high에서는 악화됐다. 따라서 `모든 backbone·모든 열화에서 PSNR과 LPIPS가 동시에 개선된다`는 주장은 하지 않는다.
4. 서로 다른 확인은행과 backbone별 route를 사용했으므로 수치 크기로 backbone 우열을 판단하지 않는다.
5. 실제 4터널에는 registered paired source reference가 없으므로 PSNR·LPIPS는 N/A다. 현장 효능은 AUPRC·matched FPR·apparent width로 평가한다.

## 1.5 재현·검증

- C010 pooled PSNR 동결값 재현 최대 절대차: `3.9424363×10^-9 dB`
- C039 summary SHA-256: `0fc7ef57c1b66ac243e7272aecc0a00250dc892100ec9305ca9e0422b89aa342`
- C049 summary SHA-256: `09e0215302d7ce429d3c98f0af84c135c58fbc42ecd213846de204a09f4dd2a9`
- NAFNet summary SHA-256: `f6eb76e91b7c6e58a23f0bf1f047bcc65104346e32ea78743a423e7f35148d3e`
- Table 5 JSON SHA-256: `1321b2f5d645db150799516456b505b30bd1d835c08588239cb2e8d87ccc73fd`
- Table CSV SHA-256: `2e1997ec6cdff8fc5b8baa84c4b0997f4ff6e8feddbab5b1c854b977cefb8cc5`
- C060 시험: `5 passed`

LPIPS는 grayscale 출력을 RGB 세 채널로 반복하고 `[-1,1]`로 변환한 뒤 AlexNet LPIPS로 계산했다. 이 값은 source-reference feature distance이며 자연 장면의 진실성 또는 안전성을 보증하지 않는다.
