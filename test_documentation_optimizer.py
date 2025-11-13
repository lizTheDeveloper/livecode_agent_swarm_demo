"""
Tests for the documentation optimizer system
"""

import pytest
import tempfile
import os
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock
from documentation_optimizer import (
    read_repository,
    repo_reader,
    documentation_writer,
    documentation_reviewer,
    route_documentation,
    State,
    DocumentationReview,
    generate_documentation,
)


@pytest.fixture
def sample_repo(tmp_path):
    """Create a sample repository structure for testing"""
    repo_dir = tmp_path / "test_repo"
    repo_dir.mkdir()
    
    # Create some sample files
    (repo_dir / "README.md").write_text("# Test Project\n\nA test project.")
    (repo_dir / "main.py").write_text("def main():\n    print('Hello, World!')\n")
    (repo_dir / "utils.py").write_text("def helper():\n    return True\n")
    
    # Create a subdirectory
    (repo_dir / "src").mkdir()
    (repo_dir / "src" / "module.py").write_text("class TestClass:\n    pass\n")
    
    return str(repo_dir)


@pytest.fixture
def mock_llm():
    """Mock LLM for testing"""
    mock = Mock()
    mock_response = Mock()
    mock_response.content = "Sample documentation content"
    mock.invoke.return_value = mock_response
    return mock


class TestReadRepository:
    """Tests for read_repository function"""
    
    def test_read_repository_success(self, sample_repo):
        """Test reading a valid repository"""
        summary = read_repository(sample_repo)
        
        assert "test_repo" in summary
        assert "Total files analyzed" in summary
        assert "README.md" in summary or "main.py" in summary
    
    def test_read_repository_nonexistent_path(self):
        """Test reading a non-existent repository path"""
        with pytest.raises(ValueError, match="Repository path does not exist"):
            read_repository("/nonexistent/path")
    
    def test_read_repository_excludes_directories(self, tmp_path):
        """Test that excluded directories are not read"""
        repo_dir = tmp_path / "test_repo"
        repo_dir.mkdir()
        
        # Create excluded directories
        (repo_dir / ".git").mkdir()
        (repo_dir / "__pycache__").mkdir()
        (repo_dir / "node_modules").mkdir()
        
        # Create a valid file
        (repo_dir / "test.py").write_text("test")
        
        summary = read_repository(str(repo_dir))
        
        # Should not include excluded directories
        assert ".git" not in summary
        assert "__pycache__" not in summary
        assert "node_modules" not in summary
        assert "test.py" in summary
    
    def test_read_repository_limits_file_size(self, tmp_path):
        """Test that large files are skipped"""
        repo_dir = tmp_path / "test_repo"
        repo_dir.mkdir()
        
        # Create a large file (>100KB)
        large_file = repo_dir / "large.txt"
        large_file.write_text("x" * 200000)
        
        # Create a normal file
        (repo_dir / "normal.py").write_text("normal")
        
        summary = read_repository(str(repo_dir))
        
        # Large file should be skipped
        assert "large.txt" not in summary or "large.txt" not in summary.split("\n\nCode content:")[1]
        assert "normal.py" in summary


class TestRepoReader:
    """Tests for repo_reader function"""
    
    def test_repo_reader_success(self, sample_repo):
        """Test repo_reader with valid state"""
        state: State = {
            "repo_path": sample_repo,
            "code_summary": "",
            "documentation": "",
            "feedback": "",
            "review_status": "",
            "revision_round": 0,
        }
        
        result = repo_reader(state)
        
        assert "code_summary" in result
        assert len(result["code_summary"]) > 0
    
    def test_repo_reader_default_path(self, tmp_path):
        """Test repo_reader with default path"""
        # Create a minimal repo in current directory
        test_file = tmp_path / "test.txt"
        test_file.write_text("test")
        
        state: State = {
            "repo_path": str(tmp_path),
            "code_summary": "",
            "documentation": "",
            "feedback": "",
            "review_status": "",
            "revision_round": 0,
        }
        
        result = repo_reader(state)
        assert "code_summary" in result


class TestDocumentationWriter:
    """Tests for documentation_writer function"""
    
    @patch('documentation_optimizer.llm')
    def test_documentation_writer_initial(self, mock_llm, sample_repo):
        """Test documentation writer on initial run"""
        mock_response = Mock()
        mock_response.content = "Generated documentation"
        mock_llm.invoke.return_value = mock_response
        
        state: State = {
            "repo_path": sample_repo,
            "code_summary": "Sample code summary",
            "documentation": "",
            "feedback": "",
            "review_status": "",
            "revision_round": 0,
        }
        
        result = documentation_writer(state)
        
        assert "documentation" in result
        assert result["documentation"] == "Generated documentation"
        assert result["revision_round"] == 1
        mock_llm.invoke.assert_called_once()
    
    @patch('documentation_optimizer.llm')
    def test_documentation_writer_with_feedback(self, mock_llm):
        """Test documentation writer with feedback from previous round"""
        mock_response = Mock()
        mock_response.content = "Revised documentation"
        mock_llm.invoke.return_value = mock_response
        
        state: State = {
            "repo_path": ".",
            "code_summary": "Sample code summary",
            "documentation": "Previous documentation",
            "feedback": "Add more examples",
            "review_status": "needs_revision",
            "revision_round": 1,
        }
        
        result = documentation_writer(state)
        
        assert "documentation" in result
        assert result["revision_round"] == 2
        # Should include feedback in the prompt
        call_args = mock_llm.invoke.call_args[0][0]
        assert "feedback" in call_args.lower() or "Add more examples" in call_args


class TestDocumentationReviewer:
    """Tests for documentation_reviewer function"""
    
    @patch('documentation_optimizer.reviewer_invoke')
    def test_documentation_reviewer_approved(self, mock_reviewer):
        """Test reviewer approving documentation"""
        mock_review = DocumentationReview(
            status="approved",
            feedback="Great documentation!",
            quality_score=9
        )
        mock_reviewer.return_value = mock_review
        
        state: State = {
            "repo_path": ".",
            "code_summary": "Code summary",
            "documentation": "Sample documentation",
            "feedback": "",
            "review_status": "",
            "revision_round": 1,
        }
        
        result = documentation_reviewer(state)
        
        assert result["review_status"] == "approved"
        assert result["feedback"] == "Great documentation!"
    
    @patch('documentation_optimizer.reviewer_invoke')
    def test_documentation_reviewer_needs_revision(self, mock_reviewer):
        """Test reviewer requesting revision"""
        mock_review = DocumentationReview(
            status="needs_revision",
            feedback="Add more examples",
            quality_score=6
        )
        mock_reviewer.return_value = mock_review
        
        state: State = {
            "repo_path": ".",
            "code_summary": "Code summary",
            "documentation": "Sample documentation",
            "feedback": "",
            "review_status": "",
            "revision_round": 1,
        }
        
        result = documentation_reviewer(state)
        
        assert result["review_status"] == "needs_revision"
        assert "examples" in result["feedback"].lower()


class TestRouteDocumentation:
    """Tests for route_documentation function"""
    
    def test_route_approved(self):
        """Test routing when documentation is approved"""
        state: State = {
            "repo_path": ".",
            "code_summary": "",
            "documentation": "",
            "feedback": "",
            "review_status": "approved",
            "revision_round": 1,
        }
        
        route = route_documentation(state)
        assert route == "Accepted"
    
    def test_route_needs_revision(self):
        """Test routing when documentation needs revision"""
        state: State = {
            "repo_path": ".",
            "code_summary": "",
            "documentation": "",
            "feedback": "",
            "review_status": "needs_revision",
            "revision_round": 1,
        }
        
        route = route_documentation(state)
        assert route == "Needs_Revision"
    
    def test_route_max_rounds_reached(self):
        """Test routing when max rounds are reached"""
        state: State = {
            "repo_path": ".",
            "code_summary": "",
            "documentation": "",
            "feedback": "",
            "review_status": "needs_revision",
            "revision_round": 5,
        }
        
        route = route_documentation(state)
        assert route == "Max_Rounds_Reached"
    
    def test_route_max_rounds_exceeded(self):
        """Test routing when rounds exceed max"""
        state: State = {
            "repo_path": ".",
            "code_summary": "",
            "documentation": "",
            "feedback": "",
            "review_status": "needs_revision",
            "revision_round": 10,
        }
        
        route = route_documentation(state)
        assert route == "Max_Rounds_Reached"


class TestGenerateDocumentation:
    """Integration tests for generate_documentation"""
    
    @patch('documentation_optimizer.documentation_workflow')
    def test_generate_documentation_success(self, mock_workflow, sample_repo):
        """Test successful documentation generation"""
        mock_final_state: State = {
            "repo_path": sample_repo,
            "code_summary": "Code summary",
            "documentation": "Final documentation",
            "feedback": "",
            "review_status": "approved",
            "revision_round": 2,
        }
        mock_workflow.invoke.return_value = mock_final_state
        
        result = generate_documentation(repo_path=sample_repo, show_graph=False)
        
        assert result["documentation"] == "Final documentation"
        assert result["review_status"] == "approved"
        assert result["revision_round"] == 2
        mock_workflow.invoke.assert_called_once()
    
    @patch('documentation_optimizer.documentation_workflow')
    def test_generate_documentation_with_error(self, mock_workflow):
        """Test error handling in documentation generation"""
        mock_workflow.invoke.side_effect = Exception("Test error")
        
        with pytest.raises(Exception, match="Test error"):
            generate_documentation(repo_path=".", show_graph=False)


class TestStateStructure:
    """Tests for State TypedDict structure"""
    
    def test_state_required_fields(self):
        """Test that State has all required fields"""
        state: State = {
            "repo_path": ".",
            "code_summary": "",
            "documentation": "",
            "feedback": "",
            "review_status": "",
            "revision_round": 0,
        }
        
        assert isinstance(state["repo_path"], str)
        assert isinstance(state["code_summary"], str)
        assert isinstance(state["documentation"], str)
        assert isinstance(state["feedback"], str)
        assert isinstance(state["review_status"], str)
        assert isinstance(state["revision_round"], int)


class TestDocumentationReview:
    """Tests for DocumentationReview Pydantic model"""
    
    def test_documentation_review_approved(self):
        """Test creating an approved review"""
        review = DocumentationReview(
            status="approved",
            feedback="Excellent work!",
            quality_score=9
        )
        
        assert review.status == "approved"
        assert review.feedback == "Excellent work!"
        assert review.quality_score == 9
    
    def test_documentation_review_needs_revision(self):
        """Test creating a review that needs revision"""
        review = DocumentationReview(
            status="needs_revision",
            feedback="Add more details",
            quality_score=6
        )
        
        assert review.status == "needs_revision"
        assert review.quality_score == 6
    
    def test_documentation_review_quality_score_validation(self):
        """Test quality score validation"""
        # Valid scores
        review1 = DocumentationReview(
            status="approved",
            feedback="Good",
            quality_score=1
        )
        assert review1.quality_score == 1
        
        review2 = DocumentationReview(
            status="approved",
            feedback="Good",
            quality_score=10
        )
        assert review2.quality_score == 10
        
        # Invalid scores should raise validation error
        with pytest.raises(Exception):  # Pydantic validation error
            DocumentationReview(
                status="approved",
                feedback="Good",
                quality_score=0  # Below minimum
            )
        
        with pytest.raises(Exception):  # Pydantic validation error
            DocumentationReview(
                status="approved",
                feedback="Good",
                quality_score=11  # Above maximum
            )

