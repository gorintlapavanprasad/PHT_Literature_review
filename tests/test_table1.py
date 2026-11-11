import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import make_table1 as M  # noqa: E402


def test_table1_has_8_questions_with_5_distinct_options_each():
    rows = M.build()
    assert len(rows) == 8
    for title, opts in rows:
        assert [o[0] for o in opts] == [1, 2, 3, 4, 5], title
        assert len({o[1] for o in opts}) == 5, title          # the manuscript repeated option 3 as 4 and 5
    assert "Solo vs. Social" in [t for t, _ in rows] and "Solo vs. Solo" not in [t for t, _ in rows]


def test_table1_outputs_render():
    rows = M.build()
    md, tex = M.to_markdown(rows), M.to_latex(rows)
    assert md.count("(5)") == 8 and tex.count(r"\addlinespace") == 8
    assert "entirely asymmetrical" in md.lower()
