# Contributing to CIB-Med-1

We welcome contributions to CIB-Med-1! This document provides guidelines for contributing to the project.

## Getting Started

### Development Setup

1. **Fork and clone the repository**
   ```bash
   git clone https://github.com/your-username/CIB-Med-1.git
   cd CIB-Med-1
   ```

2. **Create a virtual environment**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install in development mode**
   ```bash
   pip install -e ".[dev]"
   ```

4. **Run tests to verify setup**
   ```bash
   pytest tests/ -v
   ```

## Development Workflow

### Code Style

We use the following tools for code quality:

- **Black** for code formatting
- **isort** for import sorting
- **mypy** for type checking

Run formatting before committing:
```bash
black cib_med/ tests/
isort cib_med/ tests/
mypy cib_med/
```

### Testing

- Write tests for new features
- Ensure all tests pass before submitting PR
- Aim for high test coverage

```bash
# Run all tests
pytest tests/ -v

# Run with coverage
pytest tests/ --cov=cib_med --cov-report=html

# Run specific test file
pytest tests/test_metrics.py -v
```

### Documentation

- Add docstrings to all public functions and classes
- Follow NumPy docstring format
- Update README.md for significant changes

## Types of Contributions

### Bug Reports

When reporting bugs, please include:
- Python version and OS
- Steps to reproduce
- Expected vs actual behavior
- Relevant error messages

### Feature Requests

For new features:
- Describe the use case
- Explain how it relates to the paper
- Consider backward compatibility

### Pull Requests

1. **Create a feature branch**
   ```bash
   git checkout -b feature/your-feature-name
   ```

2. **Make your changes**
   - Follow code style guidelines
   - Add tests for new functionality
   - Update documentation

3. **Commit with clear messages**
   ```bash
   git commit -m "Add feature: description of change"
   ```

4. **Push and create PR**
   ```bash
   git push origin feature/your-feature-name
   ```

5. **PR Guidelines**
   - Link to related issues
   - Describe changes clearly
   - Include test results
   - Request review from maintainers

## Code Organization

```
cib_med/
├── core/           # Core components (coordinates, evaluator, trajectory)
├── metrics/        # Evaluation metrics
├── guidance/       # Diffusion guidance methods
├── baselines/      # Baseline editing methods
├── analysis/       # Analysis tools (ablation, correlation, Pareto)
├── visualization/  # Plotting and figure generation
├── models/         # Model architectures
├── data/           # Data loading utilities
└── utils/          # General utilities
```

### Adding New Components

**New Metric:**
1. Add to `cib_med/metrics/`
2. Follow existing metric interface
3. Add tests in `tests/test_metrics.py`
4. Document in README

**New Guidance Method:**
1. Inherit from `GuidanceMethod` base class
2. Implement `compute_guidance()` method
3. Add to `cib_med/guidance/__init__.py`
4. Add tests

**New Baseline:**
1. Inherit from `BaseEditor`
2. Implement `edit()` method
3. Add to `cib_med/baselines/__init__.py`

## Paper Alignment

When contributing, ensure changes align with the NeurIPS paper:

- **Section 2**: Semantic coordinate system
- **Section 3**: Trajectory-level metrics
- **Section 4**: Constrained diffusion guidance
- **Section 5**: Experimental setup
- **Section 6**: Correlation analysis
- **Section 7**: Synthetic stress tests

## Questions?

Open an issue for questions about:
- Implementation details
- Paper clarifications
- Feature discussions

## License

By contributing, you agree that your contributions will be licensed under the MIT License.

Thank you for contributing to CIB-Med-1!
