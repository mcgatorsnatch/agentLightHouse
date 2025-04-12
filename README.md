# BioLiteAgent

**A Lightweight, Bio-Inspired Task Management Agent**

![Python](https://img.shields.io/badge/python-3.8%2B-blue) ![License](https://img.shields.io/badge/license-MIT-green) ![Status](https://img.shields.io/badge/status-alpha-orange)

BioLiteAgent is a Python-based framework for hierarchical task planning and execution, inspired by biological cognition. It combines a robust task decomposition system with memory management, error handling, and adaptive replanning to tackle complex goals efficiently. Whether you're automating workflows, prototyping AI agents, or exploring bio-inspired computing, BioLiteAgent offers a flexible and extensible foundation.

---

## Table of Contents

- [Overview](#overview)
- [Key Features](#key-features)
- [Installation](#installation)
- [Usage](#usage)
- [Testing](#testing)
- [Contributing](#contributing)
- [License](#license)
- [Contact](#contact)

---

## Overview

BioLiteAgent models intelligent task management through two core components:

- **TaskNode**: Represents tasks in a hierarchical structure with priorities, dependencies, and metadata for flexible planning.
- **BioLiteAgent**: A lightweight agent that decomposes goals into subtasks, executes them with tools, manages memory, and adapts to failures or stagnation using fast/slow cognitive loops.

Inspired by biological systems, the agent uses semantic memory, temporal awareness, and concept drift detection to maintain robust performance. It’s designed for developers and researchers who need a modular, testable system for task automation or AI experimentation.

---

## Key Features

- **Hierarchical Task Planning**:
  - Organizes tasks with priorities (0-1 scale) and dependencies.
  - Supports dynamic subtask creation and execution tracking.

- **Advanced Task Decomposition**:
  - Employs multiple strategies (conjunctions, steps, components, intent, patterns).
  - Evaluates subtasks based on specificity, count, and coverage for optimal breakdowns.

- **Robust Execution**:
  - Executes tasks with retries (up to 3 attempts) and exponential backoff.
  - Uses fallback tools to handle failures gracefully.

- **Memory Management**:
  - Maintains a capped memory queue (50 items) with temporal decay.
  - Implements semantic caching for context-aware decisions.

- **Adaptive Replanning**:
  - Detects stagnation and concept drift to trigger alternative plans.
  - Validates tasks to prune irrelevant or redundant ones.

- **Comprehensive Testing**:
  - Includes a test suite covering decomposition, execution, memory, priorities, dependencies, drift, and retries.

- **Customizable Personality**:
  - Configurable agent personality (e.g., "Witty and helpful") for tailored interactions.

---

## Installation

### Prerequisites
- Python 3.8 or higher
- No external dependencies required (standard library only)

### Steps
1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/BioLiteAgent.git
   cd BioLiteAgent
   ```

2. (Optional) Create a virtual environment:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. Place the `BioLiteAgent.py` file in your project directory.

---

## Usage

BioLiteAgent is easy to integrate into your projects. Below is a quick example to get started.

### Example: Planning a Trip
```python
from BioLiteAgent import BioLiteAgent, TaskNode

# Define tools
tools = {
    "say": lambda x: x,
    "book": lambda x: f"Booked: {x.split('book ')[-1]}",
    "plan": lambda x: f"Planned: {x.split('plan ')[-1]}",
    "search": lambda x: f"Searched for: {x.split('search ')[-1]}",
    "remind": lambda x: f"Set reminder: {x.split('remind ')[-1]}"
}

# Initialize agent
agent = BioLiteAgent(tools, personality="Witty and helpful")

# Register fallback tools
agent.register_fallback("book", lambda x: f"Alternative booking: {x.split('book ')[-1]}")
agent.register_fallback("default", lambda x: f"I'll try a different approach: {x}")

# Run a task
results = agent.run("Plan a trip and book a flight", steps=125)

# Print results
print("\nFirst 5 steps:")
for r in results[:5]:
    print(json.dumps(r, indent=2))

# Display summary
completed = sum(1 for r in results if r.get("status") == "executed")
planned = sum(1 for r in results if r.get("status") == "planned")
print(f"\nSummary: {len(results)} steps, {completed} tasks executed, {planned} planning steps")
```

### Output
The agent will decompose the goal into subtasks (e.g., "Research destinations", "Book flights"), execute them, and adapt to any failures, producing a detailed log of planning and execution steps.

---

## Testing

BioLiteAgent includes a comprehensive test suite to ensure reliability. Run it to verify core functionalities:

```python
test_results = agent.run_test_suite()
print(f"Test suite overall score: {test_results['overall_score']:.2f}")
```

The suite tests:
- **Decomposition**: Ensures tasks break down into meaningful subtasks.
- **Execution**: Validates retries and fallbacks.
- **Memory**: Checks context retrieval accuracy.
- **Priority**: Confirms dynamic priority adjustments.
- **Dependencies**: Verifies dependency handling.
- **Drift Recovery**: Tests adaptation to concept drift.
- **Retries**: Ensures failure handling.

---

## Contributing

We welcome contributions to make BioLiteAgent even better! Here's how to get involved:

1. **Fork the Repository**:
   - Click the "Fork" button on GitHub and clone your fork:
     ```bash
     git clone https://github.com/yourusername/BioLiteAgent.git
     ```

2. **Create a Branch**:
   ```bash
   git checkout -b feature/your-feature-name
   ```

3. **Make Changes**:
   - Add new features, fix bugs, or improve documentation.
   - Ensure code follows PEP 8 style guidelines.

4. **Run Tests**:
   - Verify your changes pass the test suite.

5. **Submit a Pull Request**:
   - Push your branch and create a PR with a clear description of changes.

Please read our [CONTRIBUTING.md](CONTRIBUTING.md) (coming soon) for detailed guidelines.

---

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.
