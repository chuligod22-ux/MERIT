#!/usr/bin/env python3
"""Synchronise the confirmed C061 motion result across readable manuscripts."""

from __future__ import annotations

import re
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]


ENGLISH = {
    ROOT / "manuscript_r3_6" / "main_readable_pre_r2_4.md": "Table 10",
    ROOT / "manuscript_r3_6" / "main_reviewer_highlight.md": "Table 10",
    ROOT / "manuscript_r2_4" / "main_readable.md": "Table 7",
}
KOREAN = {
    ROOT / "manuscript_r3_6" / "main_readable_pre_r2_4_ko.md": "표 10",
    ROOT / "manuscript_r3_6" / "main_reviewer_highlight_ko.md": "표 10",
    ROOT / "manuscript_r2_4" / "main_readable_ko.md": "표 7",
}


EN_METHOD = (
    "The profiles used parameter-conditioned inverse transforms for illumination, "
    "saturation, and veiling, bounded spatial unsharp filtering for space-varying blur, "
    "and a Poisson–Gaussian-trained route with a fixed mild-noise fusion. For the supplied "
    "linear-motion kernels, the final profile used non-blind regularised Wiener deconvolution "
    "($K=0.003$) at lengths 3 and 7 pixels and 32-step Landweber deconvolution (step 1.5) "
    "at length 13 pixels. The three motion routes were selected from 128 bounded candidates "
    "per severity using 50 crops from 33 physical parents, then the code, parameters, source "
    "bank, and gates were frozen before a single route-specific evaluation on 98 crops from "
    "68 physical parents with zero parent overlap. The other 15 family–severity cells retain "
    "the previously frozen confirmation outputs. Parent-cluster bootstrap confidence intervals "
    "used 10,000 replicates. The final comparator is the unrestored degraded input. Because the "
    "true synthetic PSF length and angle are supplied, this protocol evaluates known-operator "
    "recovery rather than blind field deblurring."
)

KO_METHOD = (
    "프로파일에는 조명, 포화 및 베일링에 대한 매개변수 조건부 역변환, 공간가변 블러에 "
    "대한 제한된 공간 언샤프 필터, 고정 약노이즈 융합을 포함한 푸아송–가우시안 학습 "
    "경로를 사용했다. 제공된 선형 모션 커널에는 길이 3 및 7픽셀에서 비맹목 정규화 "
    "Wiener 역필터($K=0.003$), 길이 13픽셀에서 32회 Landweber 역필터(스텝 1.5)를 "
    "사용했다. 모션 경로는 강도별 128개 제한 후보를 물리적 상위 구간 33개의 크롭 "
    "50개에서 선택한 뒤 코드, 매개변수, 원본 뱅크 및 게이트를 고정했다. 이후 상위 구간 "
    "중복이 전혀 없는 물리적 상위 구간 68개의 크롭 98개에서 경로별 1회 확증했다. 나머지 "
    "15개 계열–강도 셀은 앞서 고정한 확증 출력을 유지한다. 상위 구간 군집 부트스트랩 "
    "신뢰구간에는 10,000회 반복을 사용했고 최종 비교 대상은 복원하지 않은 열화 입력이다. "
    "실제 합성 PSF의 길이와 각도를 제공하므로 이 시험은 알려진 연산자의 복원을 평가하며, "
    "현장 영상의 맹목 모션 디블러링을 평가하지 않는다."
)


def sub_once(text: str, pattern: str, replacement: str, label: str) -> str:
    updated, count = re.subn(pattern, replacement, text, count=1, flags=re.DOTALL)
    if count != 1:
        raise RuntimeError(f"Expected one replacement for {label}, found {count}")
    return updated


def update_english(path: Path, table: str) -> None:
    text = path.read_text(encoding="utf-8")
    text = sub_once(
        text,
        r"The profiles used parameter-conditioned inverse transforms.*?it did not change the frozen confirmation verdict\.",
        EN_METHOD,
        f"{path}: methods",
    )
    result = (
        f"All 18 final family–severity cells improved relative to the unrestored degraded input "
        f"under the paired PSNR, LPIPS, gradient-error, and overshoot criteria ({table}). The "
        "15 non-motion cells retain their previously frozen confirmation outputs. For linear "
        "motion, the separately frozen known-kernel route passed on all three severities in the "
        "98-crop, 68-parent confirmation cohort: the PSNR changes were $+10.006906$, $+6.702691$, "
        "and $+3.618613$ dB, and the LPIPS changes were $-0.066200$, $-0.179685$, and $-0.143313$. "
        "All corresponding parent-bootstrap confidence intervals favoured the route, and no final "
        "cell was mixed or worse."
    )
    text = sub_once(
        text,
        r"Against the original single restoration route, all 18 frozen family[–-]severity confirmation cells.*?no cell was mixed or worse under the audit rule \(Table (?:10|7)\)\.",
        result,
        f"{path}: result",
    )
    text, table_count = re.subn(
        r"\| Linear-motion PSF \| zero-gain exact bypass \| 3/3 preserved \| \$0\.000\$ \| \$0\.000\$ \|",
        "| Linear-motion PSF | known-kernel Wiener/Landweber | 3/3 improved | $+3.619$ to $+10.007$ | $-0.179685$ to $-0.066200$ |",
        text,
    )
    if table_count != 1:
        raise RuntimeError(f"Expected one motion table row in {path}, found {table_count}")
    text = sub_once(
        text,
        r"Among the 15 input-improvement cells, the smallest PSNR improvement.*?not field-performance expectations\.",
        (
            "Across all 18 input-improvement cells, the smallest PSNR improvement occurred for "
            "severity-3 linear motion ($+3.618613$ dB; 95% CI, $+3.527309$ to $+3.719351$), and "
            "the smallest LPIPS reduction occurred for severity-1 saturation/clipping "
            "($-0.005641$; 95% CI, $-0.005720$ to $-0.005567$). The motion profile improved all "
            "three supplied-PSF conditions, while gradient error and edge overshoot also decreased. "
            "These results establish synthetic non-blind recovery only: they do not show that MERIT "
            "can infer an unknown motion kernel in a tunnel image. The very large input-referenced "
            "gains for illumination, saturation, and veiling likewise arise from deterministic "
            "inversion with known synthetic parameters and are not field-performance expectations."
        ),
        f"{path}: weakest cell",
    )
    text = text.replace(
        "all 18 frozen family–severity cells",
        "all 18 final family–severity cells",
    )
    text = text.replace(
        "The linear-motion panel is an exact input bypass and makes no deblurring claim.",
        "The linear-motion panel uses the frozen non-blind route with the supplied synthetic PSF; it is not evidence of blind field deblurring.",
    )
    text = text.replace(
        "The linear-motion route is an exact bypass that prevents additional harm but does not deblur the input. In addition, the candidate-versus-input classification was added after the frozen confirmation as a descriptive interpretation and is not a second independent confirmation.",
        "The linear-motion result likewise uses the supplied synthetic PSF in a frozen non-blind Wiener/Landweber route. It does not demonstrate unknown-kernel estimation or motion deblurring in real tunnel imagery. The 68-parent cohort is a route-specific held-out confirmation for this motion profile, not a newly acquired field cohort.",
    )
    text = text.replace(
        "In the deliberately perturbed synthetic battery, 15 of 18 family–severity cells improved relative to the unrestored input and the three linear-motion cells preserved that input exactly; no audited cell worsened. This supports parameter-conditioned robustness within the six tested synthetic families. The motion result is a fail-safe preservation outcome rather than evidence of deblurring, and the battery does not establish blind degradation estimation or sensor-calibrated field recovery.",
        "In the deliberately perturbed synthetic battery, all 18 family–severity cells improved relative to the unrestored input and no final cell worsened. The known-kernel motion route improved PSNR by $+10.006906$, $+6.702691$, and $+3.618613$ dB across the three tested severities, with favourable LPIPS, gradient-error, and overshoot intervals. This supports parameter-conditioned robustness within the six tested synthetic families, while the supplied-PSF protocol does not establish blind degradation estimation, real-field motion deblurring, or sensor-calibrated field recovery.",
    )
    path.write_text(text, encoding="utf-8", newline="\n")


def update_korean(path: Path, table: str) -> None:
    text = path.read_text(encoding="utf-8")
    text = sub_once(
        text,
        r"프로파일에는 조명, 포화 및 베일링에 대한 매개변수 조건부 역변환.*?고정된 확증 판정을 바꾸지는 않았다\.",
        KO_METHOD,
        f"{path}: methods",
    )
    result = (
        f"최종 계열–강도 셀 18개 모두가 대응 PSNR, LPIPS, 그래디언트 오차 및 오버슈트 "
        f"기준에서 복원하지 않은 열화 입력보다 개선되었다({table}). 비모션 셀 15개는 앞서 "
        "고정한 확증 출력을 유지한다. 선형 모션은 별도로 고정한 알려진-커널 경로가 크롭 "
        "98개·물리적 상위 구간 68개의 확증군에서 3개 강도를 모두 통과했다. PSNR 변화는 "
        "$+10.006906$, $+6.702691$, $+3.618613$ dB였고 LPIPS 변화는 $-0.066200$, "
        "$-0.179685$, $-0.143313$이었다. 모든 상위 구간 부트스트랩 신뢰구간이 개선 방향이었고 "
        "혼합 또는 악화된 최종 셀은 없었다."
    )
    text = sub_once(
        text,
        r"기존 단일 복원 경로와 비교했을 때 고정된 계열[–-]강도 확증 셀 18개.*?감사 규칙에서 혼합되거나 악화된 셀은 없었다\(표 (?:10|7)\)\.",
        result,
        f"{path}: result",
    )
    text, table_count = re.subn(
        r"\| 선형 모션 PSF \| 영이득 정확 우회 \| 3/3 보존 \| \$0\.000\$ \| \$0\.000\$ \|",
        "| 선형 모션 PSF | 알려진-커널 Wiener/Landweber | 3/3 개선 | $+3.619$~$+10.007$ | $-0.179685$~$-0.066200$ |",
        text,
    )
    if table_count != 1:
        raise RuntimeError(f"Expected one Korean motion table row in {path}, found {table_count}")
    text = sub_once(
        text,
        r"입력보다 개선된 15개 셀 중 가장 작은 PSNR 개선.*?현장 성능의 예상값으로 볼 수 없다\.",
        (
            "입력보다 개선된 18개 셀 전체에서 가장 작은 PSNR 개선은 강도 3 선형 모션의 "
            "$+3.618613$ dB였다(95% CI, $+3.527309$~$+3.719351$). 가장 작은 LPIPS 감소는 "
            "강도 1 포화·클리핑의 $-0.005641$이었다(95% CI, $-0.005720$~$-0.005567$). "
            "모션 프로파일은 제공된 PSF의 3개 조건을 모두 개선했고 그래디언트 오차와 엣지 "
            "오버슈트도 감소했다. 이 결과는 합성 비맹목 복원만 입증하며, MERIT이 터널 영상의 "
            "미지 모션 커널을 추정할 수 있음을 뜻하지 않는다. 조명, 포화 및 베일링의 큰 입력 "
            "기준 이득도 알려진 합성 매개변수의 결정론적 역변환에서 나온 값이며 현장 성능의 "
            "예상값으로 볼 수 없다."
        ),
        f"{path}: weakest cell",
    )
    text = text.replace("고정된 계열–강도 셀 18개 전체", "최종 계열–강도 셀 18개 전체")
    text = text.replace(
        "선형 모션 패널은 입력을 정확히 우회한 결과이며 디블러링 효능을 주장하지 않는다.",
        "선형 모션 패널은 제공된 합성 PSF를 사용하는 고정 비맹목 경로의 결과이며, 현장 맹목 디블러링의 근거가 아니다.",
    )
    text = text.replace(
        "선형 모션 경로는 추가 손상을 막는 정확 우회이지만 입력을 디블러링하지 않는다. 또한 후보–입력 분류는 고정 확증 후 설명적 해석으로 추가되었으며 두 번째 독립 확증이 아니다.",
        "선형 모션 결과도 제공된 합성 PSF를 사용하는 고정 비맹목 Wiener/Landweber 경로에서 얻었다. 이는 실제 터널 영상의 미지 커널 추정이나 모션 디블러링을 입증하지 않는다. 68개 상위 구간 코호트는 이 모션 프로파일의 경로별 미사용 확증이며 새로 촬영한 현장 코호트는 아니다.",
    )
    text = text.replace(
        "의도적으로 변형한 합성 배터리에서는 계열–강도 셀 18개 중 15개가 복원하지 않은 입력보다 개선되었고, 선형 모션 셀 3개는 입력을 정확히 보존했으며, 감사한 어느 셀도 악화되지 않았다. 이는 시험한 합성 계열 6개 안에서 매개변수 조건부 강건성을 뒷받침한다. 모션 결과는 디블러링의 근거가 아니라 안전 보존 결과이며, 이 배터리는 맹목적 열화 추정이나 센서 보정 현장 회복을 입증하지 않는다.",
        "의도적으로 변형한 합성 배터리에서는 계열–강도 셀 18개 모두가 복원하지 않은 입력보다 개선되었고 악화된 최종 셀은 없었다. 알려진-커널 모션 경로의 PSNR 개선은 세 강도에서 $+10.006906$, $+6.702691$, $+3.618613$ dB였고 LPIPS, 그래디언트 오차 및 오버슈트 신뢰구간도 유리했다. 이는 시험한 합성 계열 6개 안에서 매개변수 조건부 강건성을 뒷받침하지만, 제공된 PSF 시험은 맹목적 열화 추정, 실제 현장 모션 디블러링 또는 센서 보정 현장 회복을 입증하지 않는다.",
    )
    path.write_text(text, encoding="utf-8", newline="\n")


def main() -> None:
    for path, table in ENGLISH.items():
        update_english(path, table)
        print(f"updated {path}")
    for path, table in KOREAN.items():
        update_korean(path, table)
        print(f"updated {path}")


if __name__ == "__main__":
    main()
