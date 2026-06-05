import unittest
from hypothesis import given, strategies as st
from pydantic import ValidationError
from api.schemas import GenerateContentRequest, CreateProjectRequest

class TestAPISchemasPBT(unittest.TestCase):

    # ============================================================================
    # 1. Content Generation Request Fuzzing
    # ============================================================================
    @given(
        topic=st.text(min_size=1, max_size=1000),
        keywords=st.lists(st.text(), max_size=25),
        priority=st.text()
    )
    def test_generate_content_request_robustness(self, topic, keywords, priority):
        """
        Fuzz the GenerateContentRequest to find inputs that might crash validation
        or pass when they shouldn't.
        """
        try:
            # Attempt to create the model
            request = GenerateContentRequest(
                topic=topic,
                keywords=keywords,
                priority=priority if priority in ["low", "medium", "high", "critical"] else "medium",
                async_execution=True
            )

            # If it succeeds, verify invariants
            assert len(request.topic) >= 10, "Topic too short passed validation!"
            assert len(request.topic) <= 500, "Topic too long passed validation!"

            # Check Sanitization Invariants
            if "<script" in topic.lower():
                assert "<script" not in request.topic.lower(), "XSS script tag leaked through!"

            if request.keywords:
                assert len(request.keywords) <= 20, "Too many keywords passed!"
                for k in request.keywords:
                    assert len(k) <= 100, "Keyword too long passed!"

        except ValidationError as e:
            # Validation errors are EXPECTED for bad inputs.
            # We only care if the validation logic ITSELF crashes (uncaught exception)
            pass
        except ValueError as e:
            # Custom validators raise ValueError, which Pydantic catches.
            # If it escapes Pydantic, it's an issue, but usually wrapped.
            # Here we just catch it to ensure the TEST doesn't fail on valid rejections.
            pass

    # ============================================================================
    # 2. Edge Case: Injection Attacks
    # ============================================================================
    @given(topic=st.text())
    def test_topic_sanitization_edge_cases(self, topic):
        """Ensure sanitizer doesn't crash on any unicode input."""
        try:
            # Instantiate model to trigger full validation stack
            req = GenerateContentRequest(
                topic=topic,
                priority="medium",
                async_execution=True
            )
            # If valid, check invariants
            if req.topic:
                assert "<script>" not in req.topic
                assert "javascript:" not in req.topic
                # Also check our new invariant
                assert len(req.topic) >= 10

        except ValidationError:
            pass # Rejection is fine
        except Exception as e:
            self.fail(f"Sanitizer CRASHED on input: {repr(topic)} | Error: {e}")

    # ============================================================================
    # 3. Edge Case: Project Domain Validation
    # ============================================================================
    @given(domain=st.text())
    def test_project_domain_stripping(self, domain):
        """Ensure domain validator handles garbage gracefully."""
        try:
            req = CreateProjectRequest(name="Valid Name", domain=domain)
            if req.domain:
                assert not req.domain.startswith("http://")
                assert not req.domain.startswith("https://")
                assert not req.domain.endswith("/")
        except ValidationError:
            pass

if __name__ == '__main__':
    unittest.main()
