"""
Basic SEO Validation Module
===========================

Validates generated articles against basic SEO best practices.
"""

import re
from typing import Dict, List, Tuple, Optional
from bs4 import BeautifulSoup
from loguru import logger


class SEOValidator:
    """Professional SEO validator following industry standards.

    Implements checks for:
    - Google Search Console guidelines
    - Yoast SEO best practices
    - Moz SEO recommendations
    - Ahrefs content optimization
    """

    # SEO Thresholds (industry standards)
    TITLE_MIN_LENGTH = 30
    TITLE_MAX_LENGTH = 60
    TITLE_OPTIMAL_LENGTH = 50

    META_DESC_MIN_LENGTH = 120
    META_DESC_MAX_LENGTH = 160
    META_DESC_OPTIMAL_LENGTH = 155

    MIN_WORD_COUNT = 800  # Increased from 300 for better ranking
    OPTIMAL_WORD_COUNT = 1500

    MIN_H2_HEADINGS = 3  # Increased from 2
    KEYWORD_DENSITY_MIN = 0.5  # 0.5%
    KEYWORD_DENSITY_MAX = 2.5  # 2.5%
    OPTIMAL_KEYWORD_DENSITY = 1.0  # 1%

    MIN_INTERNAL_LINKS = 2
    MIN_EXTERNAL_LINKS = 1

    # Readability thresholds (Flesch Reading Ease)
    READABILITY_MIN = 60  # College level
    READABILITY_OPTIMAL = 70  # 8th-9th grade

    @staticmethod
    def validate_article(title: str, content: str, meta_description: str, keywords: List[str]) -> Tuple[bool, List[str]]:
        """
        Validate article SEO compliance.

        Args:
            title: Article title
            content: Article HTML content
            meta_description: SEO meta description
            keywords: Target keywords

        Returns:
            Tuple of (is_valid, list_of_issues)
        """
        issues = []

        # 1. Title validation (critical for CTR)
        if not title or len(title.strip()) == 0:
            issues.append("❌ CRITICAL: Title is empty")
        elif len(title) < SEOValidator.TITLE_MIN_LENGTH:
            issues.append(f"⚠️ Title too short ({len(title)} chars) - minimum {SEOValidator.TITLE_MIN_LENGTH} recommended")
        elif len(title) > SEOValidator.TITLE_MAX_LENGTH:
            issues.append(f"⚠️ Title too long ({len(title)} chars) - may be truncated in SERPs (max {SEOValidator.TITLE_MAX_LENGTH})")

        # Check for power words and numbers in title (boost CTR)
        if title and not any(word in title.lower() for word in ['راهنما', 'نحوه', 'چگونه', 'بهترین', '۱۰', '۵', 'کامل', 'جامع']):
            issues.append("💡 Suggestion: Add power words or numbers to title for better CTR")

        # 2. Meta description validation (affects CTR)
        if not meta_description or len(meta_description.strip()) == 0:
            issues.append("❌ CRITICAL: Meta description is empty")
        elif len(meta_description) < SEOValidator.META_DESC_MIN_LENGTH:
            issues.append(f"⚠️ Meta description too short ({len(meta_description)} chars) - minimum {SEOValidator.META_DESC_MIN_LENGTH} recommended")
        elif len(meta_description) > SEOValidator.META_DESC_MAX_LENGTH:
            issues.append(f"⚠️ Meta description too long ({len(meta_description)} chars) - will be truncated (max {SEOValidator.META_DESC_MAX_LENGTH})")

        # Check for call-to-action in meta description
        if meta_description and not any(cta in meta_description for cta in ['بیشتر بخوانید', 'کلیک کنید', 'مشاهده کنید', 'دریافت کنید']):
            issues.append("💡 Suggestion: Add call-to-action to meta description")

        # 3. Content structure validation
        if content:
            soup = BeautifulSoup(content, "html.parser")

            # Check for H1
            h1_tags = soup.find_all("h1")
            if len(h1_tags) == 0:
                issues.append("No H1 heading found")
            elif len(h1_tags) > 1:
                issues.append(f"Multiple H1 headings found ({len(h1_tags)}) - should have only one")

            # Check for H2 headings (improves scannability)
            h2_tags = soup.find_all("h2")
            if len(h2_tags) < SEOValidator.MIN_H2_HEADINGS:
                issues.append(f"⚠️ Too few H2 headings ({len(h2_tags)}) - minimum {SEOValidator.MIN_H2_HEADINGS} for proper structure")

            # Check heading hierarchy
            all_headings = soup.find_all(['h1', 'h2', 'h3', 'h4', 'h5', 'h6'])
            if len(all_headings) > 1:
                prev_level = int(all_headings[0].name[1])
                for heading in all_headings[1:]:
                    curr_level = int(heading.name[1])
                    if curr_level > prev_level + 1:
                        issues.append(f"⚠️ Heading hierarchy broken: {all_headings[0].name} to {heading.name} (skipped levels)")
                        break
                    prev_level = curr_level

            # Check word count (crucial for ranking)
            text_content = soup.get_text()
            word_count = len(text_content.split())
            if word_count < SEOValidator.MIN_WORD_COUNT:
                issues.append(f"⚠️ Content too short ({word_count} words) - minimum {SEOValidator.MIN_WORD_COUNT} for competitive ranking")
            elif word_count < SEOValidator.OPTIMAL_WORD_COUNT:
                issues.append(f"💡 Content length OK ({word_count} words) but {SEOValidator.OPTIMAL_WORD_COUNT}+ is optimal for comprehensive coverage")

            # Check for images with alt text (accessibility + SEO)
            images = soup.find_all("img")
            if len(images) == 0:
                issues.append("💡 No images found - consider adding visual content for engagement")
            images_without_alt = [img for img in images if not img.get("alt")]
            if images_without_alt:
                issues.append(f"❌ {len(images_without_alt)} images missing alt text - critical for accessibility and image SEO")

            # Check for internal and external links
            links = soup.find_all("a", href=True)
            internal_links = [a for a in links if not a['href'].startswith('http')]
            external_links = [a for a in links if a['href'].startswith('http')]

            if len(internal_links) < SEOValidator.MIN_INTERNAL_LINKS:
                issues.append(f"💡 Add more internal links ({len(internal_links)} found, {SEOValidator.MIN_INTERNAL_LINKS}+ recommended) for better site structure")
            if len(external_links) < SEOValidator.MIN_EXTERNAL_LINKS:
                issues.append(f"💡 Add authoritative external links ({len(external_links)} found) to build trust")

            # Check for lists (improves scannability)
            lists = soup.find_all(['ul', 'ol'])
            if len(lists) == 0:
                issues.append("💡 No lists found - consider adding bullet points or numbered lists for better readability")

            # Check for bold/strong text (highlights key points)
            strong_tags = soup.find_all(['strong', 'b'])
            if len(strong_tags) < 3:
                issues.append("💡 Use bold text to highlight key points (improves scannability)")

        # 4. Keyword validation
        if not keywords or len(keywords) == 0:
            issues.append("No keywords specified")
        elif len(keywords) > 10:
            issues.append(f"Too many keywords ({len(keywords)}) - maximum 10 recommended")

        # 5. Advanced keyword optimization
        if content and keywords:
            soup = BeautifulSoup(content, "html.parser")
            text_content = soup.get_text()
            text_lower = text_content.lower()
            word_count = len(text_content.split())

            primary_keyword = keywords[0] if keywords else ""
            if primary_keyword:
                keyword_count = text_lower.count(primary_keyword.lower())

                if keyword_count == 0:
                    issues.append(f"❌ CRITICAL: Primary keyword '{primary_keyword}' not found in content")
                elif word_count == 0:
                    issues.append("❌ CRITICAL: Content contains no readable text")
                else:
                    # Calculate keyword density
                    density = (keyword_count / word_count) * 100

                    if density < SEOValidator.KEYWORD_DENSITY_MIN:
                        issues.append(f"⚠️ Keyword density too low ({density:.2f}%) - aim for {SEOValidator.OPTIMAL_KEYWORD_DENSITY}%")
                    elif density > SEOValidator.KEYWORD_DENSITY_MAX:
                        issues.append(f"⚠️ Keyword stuffing detected ({density:.2f}%) - max {SEOValidator.KEYWORD_DENSITY_MAX}% recommended")

                # Check keyword in title
                if title and primary_keyword.lower() not in title.lower():
                    issues.append(f"❌ Primary keyword '{primary_keyword}' missing from title")

                # Check keyword in meta description
                if meta_description and primary_keyword.lower() not in meta_description.lower():
                    issues.append(f"⚠️ Primary keyword '{primary_keyword}' missing from meta description")

                # Check keyword in first paragraph
                paragraphs = soup.find_all('p')
                if paragraphs and primary_keyword.lower() not in paragraphs[0].get_text().lower():
                    issues.append(f"💡 Place primary keyword in first paragraph for SEO boost")

                # Check keyword in H2 headings
                h2_texts = [h2.get_text().lower() for h2 in soup.find_all('h2')]
                if not any(primary_keyword.lower() in h2 for h2 in h2_texts):
                    issues.append(f"💡 Include primary keyword in at least one H2 heading")

        is_valid = len(issues) == 0
        return is_valid, issues

    @staticmethod
    def get_seo_score(title: str, content: str, meta_description: str, keywords: List[str]) -> Dict[str, any]:
        """
        Calculate SEO score and detailed metrics.

        Returns:
            Dict with score and breakdown
        """
        score = 100
        breakdown = {}

        # Title score (20 points)
        title_score = 20
        if not title:
            title_score = 0
        elif len(title) < 30 or len(title) > 60:
            title_score = 10
        breakdown["title"] = title_score
        score -= (20 - title_score)

        # Meta description score (20 points)
        meta_score = 20
        if not meta_description:
            meta_score = 0
        elif len(meta_description) < 120 or len(meta_description) > 160:
            meta_score = 10
        breakdown["meta_description"] = meta_score
        score -= (20 - meta_score)

        # Content structure score (30 points)
        structure_score = 30
        if content:
            soup = BeautifulSoup(content, "html.parser")
            h1_count = len(soup.find_all("h1"))
            h2_count = len(soup.find_all("h2"))
            word_count = len(soup.get_text().split())

            if h1_count != 1:
                structure_score -= 10
            if h2_count < 2:
                structure_score -= 10
            if word_count < 300:
                structure_score -= 10
        else:
            structure_score = 0
        breakdown["structure"] = max(0, structure_score)
        score -= (30 - max(0, structure_score))

        # Keyword optimization score (30 points)
        keyword_score = 30
        if not keywords:
            keyword_score = 0
        elif len(keywords) > 10:
            keyword_score = 15
        breakdown["keywords"] = keyword_score
        score -= (30 - keyword_score)

        return {
            "score": max(0, score),
            "breakdown": breakdown,
            "grade": "A" if score >= 90 else "B" if score >= 75 else "C" if score >= 60 else "D" if score >= 50 else "F"
        }
