"""
Jupyter Notebook (.ipynb) 생성 유틸리티

.ipynb는 JSON 형식이므로 직접 생성 가능.
"""
import json
import os
from typing import List, Dict, Any


def make_markdown_cell(source: str) -> Dict[str, Any]:
    """마크다운 셀 생성"""
    return {
        "cell_type": "markdown",
        "metadata": {},
        "source": source.split("\n") if isinstance(source, str) else source
    }


def make_code_cell(source: str, outputs: list = None) -> Dict[str, Any]:
    """코드 셀 생성"""
    return {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": outputs or [],
        "source": source.split("\n") if isinstance(source, str) else source
    }


def create_notebook(cells: List[Dict[str, Any]], filepath: str):
    """노트북 파일 생성"""
    # 소스가 리스트일 경우 줄바꿈 처리
    processed = []
    for cell in cells:
        c = dict(cell)
        if isinstance(c["source"], list):
            lines = []
            for i, line in enumerate(c["source"]):
                if i < len(c["source"]) - 1 and not line.endswith("\n"):
                    lines.append(line + "\n")
                else:
                    lines.append(line)
            c["source"] = lines
        processed.append(c)

    nb = {
        "nbformat": 4,
        "nbformat_minor": 5,
        "metadata": {
            "kernelspec": {
                "display_name": "Python 3",
                "language": "python",
                "name": "python3"
            },
            "language_info": {
                "name": "python",
                "version": "3.11.0"
            }
        },
        "cells": processed
    }

    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    with open(filepath, "w", encoding="utf-8") as f:
        json.dump(nb, f, ensure_ascii=False, indent=1)


def problem_notebook(
    chapter_num: int,
    section_num: int,
    title: str,
    objectives: List[str],
    theory_md: str,
    guided_code: str,
    exercises: List[Dict[str, str]],
    references: List[str],
    filepath: str
):
    """문제 노트북 생성 (표준 구조)"""
    cells = []

    # 제목
    cells.append(make_markdown_cell(
        f"# Ch{chapter_num:02d}-{section_num:02d}: {title}\n"
    ))

    # 학습 목표
    obj_text = "## 학습 목표\n\n"
    for obj in objectives:
        obj_text += f"- {obj}\n"
    cells.append(make_markdown_cell(obj_text))

    # 공통 임포트
    cells.append(make_code_cell(
        "import numpy as np\n"
        "import pandas as pd\n"
        "import matplotlib.pyplot as plt\n"
        "import seaborn as sns\n"
        "from scipy import stats\n"
        "import warnings\n"
        "warnings.filterwarnings('ignore')\n"
        "\n"
        "plt.rcParams['figure.figsize'] = (10, 6)\n"
        "plt.rcParams['font.size'] = 12\n"
        "np.random.seed(42)"
    ))

    # 이론적 배경
    cells.append(make_markdown_cell(f"## 이론적 배경\n\n{theory_md}"))

    # 구현 가이드
    cells.append(make_markdown_cell("## 구현 가이드"))
    cells.append(make_code_cell(guided_code))

    # 연습 문제
    cells.append(make_markdown_cell("---\n## 연습 문제"))
    for i, ex in enumerate(exercises, 1):
        difficulty = ex.get("difficulty", "★★")
        cells.append(make_markdown_cell(
            f"### 문제 {i} [{difficulty}]\n\n{ex['description']}"
        ))
        if "hint" in ex:
            cells.append(make_markdown_cell(f"> **힌트**: {ex['hint']}"))
        cells.append(make_code_cell(ex.get("skeleton", "# 여기에 코드를 작성하세요\n")))

    # 참고 자료
    ref_text = "---\n## 참고 자료\n\n"
    for ref in references:
        ref_text += f"- {ref}\n"
    cells.append(make_markdown_cell(ref_text))

    create_notebook(cells, filepath)


def solution_notebook(
    chapter_num: int,
    section_num: int,
    title: str,
    solutions: List[Dict[str, str]],
    discussion: str,
    filepath: str
):
    """모범답안 노트북 생성 (표준 구조)"""
    cells = []

    # 제목
    cells.append(make_markdown_cell(
        f"# Ch{chapter_num:02d}-{section_num:02d}: {title} — 모범답안\n"
    ))

    # 공통 임포트
    cells.append(make_code_cell(
        "import numpy as np\n"
        "import pandas as pd\n"
        "import matplotlib.pyplot as plt\n"
        "import seaborn as sns\n"
        "from scipy import stats\n"
        "import warnings\n"
        "warnings.filterwarnings('ignore')\n"
        "\n"
        "plt.rcParams['figure.figsize'] = (10, 6)\n"
        "plt.rcParams['font.size'] = 12\n"
        "np.random.seed(42)"
    ))

    # 각 문제 풀이
    for i, sol in enumerate(solutions, 1):
        cells.append(make_markdown_cell(
            f"---\n## 문제 {i} 풀이\n\n{sol.get('approach', '')}"
        ))
        cells.append(make_code_cell(sol["code"]))
        if "interpretation" in sol:
            cells.append(make_markdown_cell(
                f"**결과 해석**: {sol['interpretation']}"
            ))

    # 확장 토론
    cells.append(make_markdown_cell(f"---\n## 확장 토론\n\n{discussion}"))

    create_notebook(cells, filepath)
