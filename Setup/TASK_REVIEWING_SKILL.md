---
name: task-creation
description: Guidelines for creating AI agent tasks. Covers required file structure, code organization, dependencies, and best practices for task.yaml, solution.sh, Dockerfile, grader.py, and test files.
license: Complete terms in LICENSE.txt
---

# Task Creation Guidelines

When creating AI agent tasks, ensure all required files are properly structured and follow these guidelines.

## Required Files

Every task must include the following files:
- `task.yaml` - Task metadata and description
- `solution.sh` - Reference solution script
- `Dockerfile` - Container environment setup
- `grader.py` - Scoring and evaluation logic
- Test files in `tests/` folder

## Task Structure

### task.yaml

**Required Components:**
- All metadata fields must be present with appropriate values
- Task description must be clear, concise, and human-written
- Task should focus on one premise/workflow, not multiple unrelated tasks
- Task description must not be ambiguous (e.g., if asking to "sort", specify ascending/descending)
- Datasets must be properly cited

**Description Guidelines:**
- Write in a natural, human-like style
- Avoid rambling or irrelevant information
- Be specific about requirements and expected behavior
- Each requirement in the description should have a corresponding test in the grader

### Dockerfile

**Requirements:**
- All libraries/packages needed to solve the task must be installed
- All dependencies must be pinned to specific versions (no unpinned dependencies)
- Should work on all target platforms (linux, mac, x86/64)

**Example:**
```dockerfile
FROM python:3.9-slim
RUN pip install pandas==1.5.3 numpy==1.24.3 scikit-learn==1.2.2
```

### solution.sh

**Requirements:**
- Must run successfully and score 100% when tested
- Must not be hardcoded - should include every step required to solve the problem
- Must not suffer from look-ahead bias (cannot use information not available to the agent)
- Must not read from or write to the `tests/` folder
- Should demonstrate the complete workflow needed to solve the task

**Structure:**
- Include all necessary steps: data loading, preprocessing, feature engineering, model training, prediction, output generation
- Use proper error handling
- Make it executable and well-commented

### grader.py

**Requirements:**
- Must have a corresponding test for each requirement mentioned in task.yaml
- Must avoid non-determinism (no random number generation, unrealistic time limits, regex parsing of outputs)
- Must return a `GradingResult` object with proper subscores and weights
- Should prevent reward hacking (agents should not be able to modify test files to achieve full reward)
- For tasks reading from "data" folder: copy input files to `tests/` folder and read from there (since data folder files could be modified)

**Structure:**
```python
def grade(task_dir: str, submission_dir: str) -> GradingResult:
    # Load test data from tests/ folder
    # Evaluate submission
    # Return GradingResult with subscores and weights
```

### Tests

**Requirements:**
- Test files should be placed in `tests/` folder
- Tests should not be brittle (avoid hardcoded strings, regex parsing)
- Hardcoded thresholds/ground truth values should be reasonable
- Tests should not assume behavior not stated in the task description
- Test functions should have docstrings
- If grader reads from "data" folder, same files must be in `tests/` folder

## Code Quality Guidelines

### Dependencies
- **Always pin dependencies** to specific versions in Dockerfile
- List all required packages and their versions
- Ensure compatibility across platforms

### Task Focus
- **Single premise/workflow**: Tasks should address one coherent problem, not multiple unrelated tasks stitched together
- Keep the scope focused and achievable

### Clarity
- **Avoid ambiguity**: Be explicit about requirements
- Example: Instead of "sort the numbers", specify "sort the numbers in ascending order"
- Make requirements testable and verifiable

### Solution Quality
- **No hardcoding**: solution.sh should solve the problem generically
- **No look-ahead bias**: Solution cannot use information that wouldn't be available to an agent
- **Complete workflow**: Include all steps from data loading to final output

### Grader Quality
- **Deterministic**: Avoid randomness, time-based checks, or regex parsing
- **Comprehensive**: Test all requirements stated in task.yaml
- **Secure**: Prevent agents from modifying test files or exploiting the grader

## File Organization

```
task_folder/
├── task.yaml          # Task metadata and description
├── solution.sh        # Reference solution
├── Dockerfile         # Environment setup
├── grader.py          # Evaluation logic
├── data/              # Training data (if needed)
│   └── train.csv
└── tests/             # Test data and ground truth
    ├── test.csv
    ├── test_ground_truth.csv
    └── (any other test files)
```

## Common Pitfalls to Avoid

- **Unpinned dependencies** - Always specify versions
- **Ambiguous descriptions** - Be explicit about requirements
- **Brittle tests** - Avoid hardcoded strings and regex parsing
- **Non-deterministic graders** - Avoid randomness and time-based checks
- **Hardcoded solutions** - solution.sh should be generic
- **Look-ahead bias** - Don't use information unavailable to agents
- **Missing test coverage** - Every requirement needs a corresponding test
- **Reward hacking vulnerabilities** - Prevent agents from modifying test files

