from execution.article_quality_gate import evaluate_article_quality


def test_counts_persian_words_after_stripping_html() -> None:
    result = evaluate_article_quality(
        "<h2>عنوان</h2><p>سلام، دنیا! مدل هوش مصنوعی.</p>",
        language="fa",
        target_word_count=900,
    )

    assert result.word_count == 6


def test_counts_persian_zwnj_compound_as_one_word() -> None:
    result = evaluate_article_quality(
        "می\u200cشود",
        language="fa",
        target_word_count=None,
        hard_minimum_words=1,
        hard_maximum_words=10,
        minimum_headings=0,
        minimum_paragraphs=0,
    )

    assert result.word_count == 1
    assert result.passed is True


def test_empty_html_structure_does_not_satisfy_release_gate() -> None:
    result = evaluate_article_quality(
        "<h2></h2><h3>&nbsp;</h3><p></p><p> </p><p>&nbsp;</p>" + "واژه " * 800,
        language="fa",
        target_word_count=900,
    )

    assert result.heading_count == 0
    assert result.paragraph_count == 0
    assert {finding.code for finding in result.findings} == {
        "insufficient_headings",
        "insufficient_paragraphs",
    }


def test_rejects_the_phase_one_short_persian_shape() -> None:
    result = evaluate_article_quality(
        "<h2>عنوان</h2><p>" + "واژه " * 22 + "</p>",
        language="fa",
        target_word_count=900,
    )

    assert result.passed is False
    assert result.word_count == 23
    assert {finding.code for finding in result.findings} == {
        "word_count_below_minimum",
        "insufficient_headings",
        "insufficient_paragraphs",
    }


def test_accepts_structured_persian_content_within_target_tolerance() -> None:
    paragraph = "واژه " * 300
    result = evaluate_article_quality(
        (
            f"<h2>مقدمه</h2><p>{paragraph}</p>"
            f"<h2>جزئیات</h2><p>{paragraph}</p>"
            f"<p>{paragraph}</p>"
        ),
        language="fa",
        target_word_count=900,
    )

    assert result.passed is True
    assert result.word_count == 902
    assert result.heading_count == 2
    assert result.paragraph_count == 3


def test_accepts_moderately_over_target_persian_seo_article() -> None:
    paragraph = "واژه " * 500
    result = evaluate_article_quality(
        (
            f"<h2>مقدمه</h2><p>{paragraph}</p>"
            f"<h2>جزئیات</h2><p>{paragraph}</p>"
            f"<h2>جمع بندی</h2><p>{paragraph}</p>"
        ),
        language="fa",
        target_word_count=900,
    )

    assert result.passed is True
    assert result.word_count == 1504
    assert result.maximum_word_count == 1800


def test_rejects_extremely_overlong_persian_article() -> None:
    paragraph = "واژه " * 635
    result = evaluate_article_quality(
        (
            f"<h2>مقدمه</h2><p>{paragraph}</p>"
            f"<h2>جزئیات</h2><p>{paragraph}</p>"
            f"<h2>جمع بندی</h2><p>{paragraph}</p>"
        ),
        language="fa",
        target_word_count=900,
    )

    assert result.passed is False
    assert result.word_count == 1909
    assert result.maximum_word_count == 1800
    assert [finding.code for finding in result.findings] == ["word_count_above_maximum"]


def test_hard_minimum_remains_valid_for_a_smaller_requested_target() -> None:
    paragraph = "واژه " * 266
    result = evaluate_article_quality(
        (
            f"<h2>مقدمه</h2><p>{paragraph}</p>"
            f"<h2>جزئیات</h2><p>{paragraph}</p>"
            f"<p>{paragraph}</p>"
        ),
        language="fa",
        target_word_count=300,
    )

    assert result.word_count == 800
    assert result.minimum_word_count == 800
    assert result.maximum_word_count == 800
    assert result.passed is True


def test_rejects_adjacent_duplicate_persian_headings() -> None:
    paragraph = "واژه " * 300
    result = evaluate_article_quality(
        (
            f"<h2>راهنمای انتخاب</h2><p>{paragraph}</p>"
            "<h2>راهنمای انتخاب</h2>"
            f"<h2>جمع بندی</h2><p>{paragraph}</p><p>{paragraph}</p>"
        ),
        language="fa",
        target_word_count=900,
    )

    assert result.passed is False
    finding = next(
        finding
        for finding in result.findings
        if finding.code == "duplicate_adjacent_headings"
    )
    assert finding.actual == "راهنمای انتخاب"


def test_requested_faq_requires_answered_questions() -> None:
    paragraph = "واژه " * 300
    result = evaluate_article_quality(
        (
            f"<h2>مقدمه</h2><p>{paragraph}</p>"
            f"<h2>جزئیات</h2><p>{paragraph}</p>"
            f"<h2>پرسش‌های متداول</h2><p>{paragraph}</p>"
        ),
        language="fa",
        target_word_count=900,
        require_faq=True,
    )

    assert result.passed is False
    assert "incomplete_required_faq" in {
        finding.code for finding in result.findings
    }


def test_requested_faq_accepts_two_answered_persian_questions() -> None:
    paragraph = "واژه " * 300
    result = evaluate_article_quality(
        (
            f"<h2>مقدمه</h2><p>{paragraph}</p>"
            f"<h2>جزئیات</h2><p>{paragraph}</p>"
            "<h2>پرسش‌های متداول</h2>"
            "<h3>پرسش نخست چیست؟</h3><p>پاسخ روشن و کاربردی نخست.</p>"
            "<h3>پرسش دوم چیست؟</h3><p>پاسخ روشن و کاربردی دوم.</p>"
            f"<p>{paragraph}</p>"
        ),
        language="fa",
        target_word_count=900,
        require_faq=True,
    )

    assert result.passed is True


def test_unrequested_faq_is_not_a_release_requirement() -> None:
    paragraph = "واژه " * 300
    result = evaluate_article_quality(
        (
            f"<h2>مقدمه</h2><p>{paragraph}</p>"
            f"<h2>جزئیات</h2><p>{paragraph}</p>"
            f"<p>{paragraph}</p>"
        ),
        language="fa",
        target_word_count=900,
    )

    assert result.passed is True
