import json
import time
import math
import random
from typing import List, Dict, Callable, Optional, Any, Union, Tuple
from collections import deque

class TaskNode:
    """Task node for hierarchical planning with enhanced metadata."""
    def __init__(self, desc: str, priority: float = 0.5, parent: Optional['TaskNode'] = None):
        self.desc = desc
        self.priority = max(0.0, min(1.0, priority))  # 0-1, higher = urgent
        self.parent = parent
        self.subtasks = []
        self.id = str(time.time())[:8]  # Rough unique ID
        self.completed = False
        self.dependencies = []  # IDs of tasks this depends on
        self.success_probability = 0.5  # Estimated chance of success
        self.is_time_sensitive = False  # Temporal importance flag
        self.metadata = {}  # Flexible metadata storage
        self.created_at = time.time()
        self.retry_count = 0  # Track retries
        self.max_retries = 3  # Maximum number of retries
        self.failed = False  # Track if execution failed after all retries

    def add_subtask(self, desc: str, priority: float = 0.5) -> 'TaskNode':
        node = TaskNode(desc, priority, self)
        self.subtasks.append(node)
        return node
    
    def add_dependency(self, task_id: str) -> None:
        """Add a dependency to this task."""
        if task_id not in self.dependencies:
            self.dependencies.append(task_id)
    
    def can_execute(self, completed_ids: List[str]) -> bool:
        """Check if all dependencies are satisfied."""
        return all(dep_id in completed_ids for dep_id in self.dependencies)

    def to_dict(self) -> Dict:
        return {
            "desc": self.desc,
            "priority": self.priority,
            "subtasks": [t.to_dict() for t in self.subtasks],
            "completed": self.completed,
            "failed": self.failed,
            "id": self.id,
            "dependencies": self.dependencies,
            "success_probability": self.success_probability,
            "is_time_sensitive": self.is_time_sensitive,
            "created_at": self.created_at,
            "retry_count": self.retry_count
        }

class BioLiteAgent:
    """Lightweight agent with fast/slow loops and memory, inspired by biological cognition."""
    def __init__(self, tools: Dict[str, Callable], personality: str = "Friendly and concise"):
        self.tools = tools  # e.g., {"say": lambda x: x, "search": lambda x: f" Searched: {x}"}
        self.memory = deque(maxlen=50)  # Recent interactions, capped for efficiency
        self.task_queue = []  # Current tasks
        self.history = []  # Long-term memory
        self.personality = personality
        self.step_count = 0
        self.last_reflection = time.time()
        self.completed_task_ids = []  # Track completed task IDs for dependencies
        self.beam_width = 3  # Number of decomposition options to consider (increased)
        self.temporal_decay_factor = 0.85  # Rate of memory decay (higher = slower decay)
        self.error_history = []  # For concept drift detection
        self.drift_threshold = 2.0  # Threshold for concept drift detection
        self.last_validation = time.time()
        self.validation_interval = 30  # Seconds between validations
        self.fallback_tools = {}  # Tools to try if primary tool fails
        self.semantic_cache = {}  # Store processed memories for better retrieval
        self.semantic_threshold = 0.6  # Threshold for semantic similarity
        self.stagnation_threshold = 5  # Number of steps without progress to trigger replan
        self.last_progress_step = 0  # Track when last meaningful progress was made
        self.max_task_age = 180  # 3 minutes maximum age for non-critical tasks
        
        # Register default fallback tool
        self.register_fallback("default", lambda x: f"{self.personality}: I'll try a different approach - {x}")

    def register_fallback(self, tool_key: str, fallback_func: Callable) -> None:
        """Register a fallback function for a specific tool."""
        self.fallback_tools[tool_key] = fallback_func
        
    def add_task(self, goal: str, priority: float = 0.5) -> TaskNode:
        """Add a top-level task and decompose it."""
        root = TaskNode(goal, priority)
        self._decompose_task(root)
        self.task_queue.append(root)
        return root

    def _decompose_task(self, task: TaskNode, depth: int = 0, max_depth: int = 4):
        """Enhanced task decomposition with multiple strategies and deeper recursion."""
        if depth >= max_depth or len(task.desc.split()) < 3:
            return
            
        # Check if the task is time-sensitive
        task.is_time_sensitive = self._is_time_sensitive(task.desc)
        
        # Try different decomposition strategies
        strategies = [
            self._decompose_by_conjunction,
            self._decompose_by_steps,
            self._decompose_by_components,
            self._decompose_by_intent,
            self._decompose_by_pattern
        ]
        
        # Try each strategy and collect results
        all_subtasks = []
        for strategy in strategies:
            subtasks = strategy(task)
            if subtasks:
                all_subtasks.append(subtasks)
        
        # If we have subtask options, evaluate and choose the best
        if all_subtasks:
            best_subtasks = self._evaluate_decompositions(all_subtasks, task)
        else:
            # Fallback: create a simple decomposition
            words = task.desc.split()
            if len(words) > 1:
                best_subtasks = [f"{words[0]} {' '.join(words[1:])}"]
            else:
                best_subtasks = [task.desc]
        
        # Clear existing subtasks and add new ones
        task.subtasks = []
        for subtask_desc in best_subtasks:
            subtask = task.add_subtask(subtask_desc, task.priority * 0.9)
            
            # Add dependencies between sequential tasks if appropriate
            if len(task.subtasks) > 1 and any(seq in subtask_desc.lower() for seq in ["after", "then", "next", "following"]):
                subtask.add_dependency(task.subtasks[-2].id)
        
        # Recurse to decompose subtasks
        for subtask in task.subtasks:
            self._decompose_task(subtask, depth + 1, max_depth)
            
    def _evaluate_decompositions(self, decomposition_options: List[List[str]], task: TaskNode) -> List[str]:
        """Evaluate decomposition options to pick the best one with improved weights."""
        # Scoring factors with adjusted weights:
        # 1. Number of subtasks (20%)
        # 2. Average specificity of subtasks (30%)
        # 3. Coverage of original task (50%) - increased to prioritize completeness
        
        scores = []
        for option in decomposition_options:
            # Factor 1: Number of subtasks (normalized to 0-1)
            num_tasks_score = min(1.0, len(option) / 10) * 0.2  # 20% weight
            
            # Factor 2: Specificity - based on average length
            specificity_scores = []
            for subtask in option:
                # Length contributes to specificity
                length_score = min(1.0, len(subtask) / 50)  # Cap at 50 chars
                specificity_scores.append(length_score)
            
            avg_specificity = sum(specificity_scores) / len(specificity_scores) if specificity_scores else 0
            specificity_factor = avg_specificity * 0.3  # 30% weight
            
            # Factor 3: Coverage - semantic similarity to original task (50% weight)
            option_text = " ".join(option)
            coverage_score = self._calculate_semantic_similarity(task.desc, option_text) * 0.5
            
            # Calculate total score
            total_score = num_tasks_score + specificity_factor + coverage_score
            scores.append(total_score)
        
        # Return the decomposition with highest score
        if not scores:
            return []
            
        best_index = scores.index(max(scores))
        return decomposition_options[best_index]

    def _decompose_by_conjunction(self, task: TaskNode) -> List[str]:
        """Decompose task by conjunctions (and, then, while)."""
        desc = task.desc
        subtasks = []
        
        # Split by various conjunctions
        for conjunction in [" and ", " then ", " while ", ", ", "; "]:
            if conjunction in desc:
                parts = desc.split(conjunction)
                return [part.strip() for part in parts]
                
        return []
    
    def _decompose_by_steps(self, task: TaskNode) -> List[str]:
        """Decompose task by identifying sequential steps."""
        words = task.desc.split()
        
        # Tasks with various action verbs can be decomposed into steps
        action_verbs = ["create", "build", "develop", "implement", "design", "plan", 
                        "analyze", "review", "test", "setup", "deploy", "organize",
                        "fix", "solve", "improve", "optimize", "refactor"]
        
        if len(words) >= 3 and words[0].lower() in action_verbs:
            # Identify what's being acted upon
            object_phrase = " ".join(words[1:])
            action = words[0].lower()
            
            # Create typical steps based on the specific action verb
            if action in ["create", "build", "develop"]:
                return [
                    f"Plan the {object_phrase}",
                    f"Design the structure of {object_phrase}",
                    f"Implement core parts of {object_phrase}",
                    f"Test the {object_phrase}",
                    f"Finalize the {object_phrase}"
                ]
            elif action in ["analyze", "review"]:
                return [
                    f"Identify key components of {object_phrase}",
                    f"Examine each component of {object_phrase}",
                    f"Document findings about {object_phrase}",
                    f"Summarize analysis of {object_phrase}"
                ]
            elif action in ["fix", "solve", "improve"]:
                return [
                    f"Diagnose issues with {object_phrase}",
                    f"Identify root causes in {object_phrase}",
                    f"Develop solutions for {object_phrase}",
                    f"Implement fixes to {object_phrase}",
                    f"Verify solutions fixed {object_phrase}"
                ]
            elif action in ["plan"]:
                return [
                    f"Research options for {object_phrase}",
                    f"Outline key milestones for {object_phrase}",
                    f"Schedule timeline for {object_phrase}",
                    f"Identify resources needed for {object_phrase}",
                    f"Document complete plan for {object_phrase}"
                ]
                
        return []
    
    def _decompose_by_components(self, task: TaskNode) -> List[str]:
        """Decompose by identifying components in the task."""
        words = task.desc.split()
        
        # Simple fallback - split into action + components if we have enough words
        if len(words) > 3:
            action = words[0]
            components = " ".join(words[1:]).split(", ")
            if len(components) > 1:
                return [f"{action} {component}" for component in components]
            else:
                # Try splitting the object into components based on domain knowledge
                object_phrase = " ".join(words[1:])
                
                # Software system components
                if any(term in object_phrase.lower() for term in ["system", "application", "platform", "software"]):
                    return [
                        f"{action} the core functionality",
                        f"{action} the user interface",
                        f"{action} the data layer",
                        f"{action} integration points",
                        f"{action} test suite",
                        f"{action} documentation"
                    ]
                
                # Business or project components
                if any(term in object_phrase.lower() for term in ["business", "project", "initiative", "program"]):
                    return [
                        f"{action} the requirements gathering",
                        f"{action} the project timeline",
                        f"{action} the resource allocation",
                        f"{action} the stakeholder communication",
                        f"{action} the risk assessment",
                        f"{action} the progress tracking"
                    ]
                    
                # Travel planning components
                if any(term in object_phrase.lower() for term in ["trip", "travel", "journey", "vacation"]):
                    return [
                        f"{action} the destination research",
                        f"{action} the transportation booking",
                        f"{action} the accommodation arrangements",
                        f"{action} the itinerary planning",
                        f"{action} the budget management"
                    ]
                    
                # Data analysis components
                if any(term in object_phrase.lower() for term in ["data", "analysis", "report", "insights"]):
                    return [
                        f"{action} the data collection",
                        f"{action} the data cleaning",
                        f"{action} the exploratory analysis",
                        f"{action} the statistical modeling",
                        f"{action} the visualization",
                        f"{action} the findings report"
                    ]
        
        return []
        
    def _decompose_by_intent(self, task: TaskNode) -> List[str]:
        """Decompose by inferring intent and generating appropriate subtasks."""
        desc = task.desc.lower()
        
        # Communication intent
        if any(term in desc for term in ["communicate", "inform", "notify", "message", "contact"]):
            return [
                "Identify key stakeholders to contact",
                "Determine the most important information to convey",
                "Draft the communication content",
                "Review and refine the message",
                "Deliver the communication",
                "Follow up on responses"
            ]
            
        # Decision making intent
        if any(term in desc for term in ["decide", "choose", "select", "determine", "pick"]):
            return [
                "Define decision criteria",
                "Gather relevant information",
                "Identify available options",
                "Evaluate options against criteria",
                "Make the decision",
                "Document decision rationale"
            ]
            
        # Learning intent
        if any(term in desc for term in ["learn", "study", "understand", "research", "investigate"]):
            return [
                "Identify key knowledge gaps",
                "Find reliable information sources",
                "Study the core concepts",
                "Practice applying the knowledge",
                "Summarize key learnings",
                "Test understanding"
            ]
            
        return []
        
    def _decompose_by_pattern(self, task: TaskNode) -> List[str]:
        """Decompose tasks based on recognized patterns in the description."""
        desc = task.desc.lower()
        
        # Pattern: "X from Y to Z" (transformation)
        if " from " in desc and " to " in desc:
            parts = desc.split(" from ")
            action = parts[0]
            sources_and_targets = parts[1].split(" to ")
            if len(sources_and_targets) >= 2:
                source = sources_and_targets[0]
                target = sources_and_targets[1]
                return [
                    f"Analyze the current state of {source}",
                    f"Define the desired state in {target}",
                    f"Identify transformation steps needed",
                    f"Execute transformation process",
                    f"Verify successful transformation to {target}"
                ]
                
        # Pattern: "X for Y" (creation for purpose)
        if " for " in desc and not " from " in desc:
            parts = desc.split(" for ")
            if len(parts) >= 2:
                creation = parts[0]
                purpose = parts[1]
                return [
                    f"Understand requirements of {purpose}",
                    f"Design {creation} to meet requirements",
                    f"Develop initial version of {creation}",
                    f"Test {creation} against purpose: {purpose}",
                    f"Refine and finalize {creation}"
                ]
                
        # Pattern: "X with Y" (using a specific tool/approach)
        if " with " in desc and not " from " in desc and not " for " in desc:
            parts = desc.split(" with ")
            if len(parts) >= 2:
                task_part = parts[0]
                tool_part = parts[1]
                return [
                    f"Learn how to use {tool_part} effectively",
                    f"Prepare resources for using {tool_part}",
                    f"Apply {tool_part} to accomplish {task_part}",
                    f"Troubleshoot any issues with {tool_part}",
                    f"Finalize {task_part} using {tool_part}"
                ]
                
        return []
    
    def _is_time_sensitive(self, desc: str) -> bool:
        """Detect if a task is time-sensitive."""
        time_indicators = ["urgent", "immediately", "soon", "today", "tomorrow", "deadline", 
                          "quickly", "asap", "promptly", "time-sensitive", "by", "schedule"]
        return any(indicator in desc.lower() for indicator in time_indicators)

    def step(self) -> Dict:
        """Fast loop: Execute or plan one step."""
        self.step_count += 1
        
        # Periodically validate our task assumptions
        if time.time() - self.last_validation > self.validation_interval:
            tasks_removed = self._validate_tasks()
            self.last_validation = time.time()
            
            # If tasks were removed due to stagnation, report this
            if tasks_removed > 0:
                return {"status": "pruned", "output": f"Removed {tasks_removed} stagnant tasks."}
            
        if not self.task_queue:
            return {"status": "idle", "output": "No tasks to process."}

        # Check for stagnation and trigger re-planning if needed
        if self.step_count - self.last_progress_step > self.stagnation_threshold:
            self.last_progress_step = self.step_count
            return self._replan_for_stagnation()

        # Pick highest-priority task that can be executed (dependencies met)
        executable_tasks = [t for t in self.task_queue 
                            if not t.completed and not t.failed and
                            (not t.subtasks or all(s.completed for s in t.subtasks)) and
                            t.can_execute(self.completed_task_ids)]
        
        if not executable_tasks:
            # Check if we have failed tasks that are blocking progress
            failed_tasks = [t for t in self.task_queue if t.failed]
            if failed_tasks:
                failed_task = failed_tasks[0]
                # Create alternative paths to work around failed tasks
                new_task = self._create_alternative_path(failed_task)
                return {"status": "replanned", "task": failed_task.desc, "new_path": new_task.desc}
            
            return {"status": "blocked", "output": "All current tasks are blocked by dependencies."}
            
        task = max(executable_tasks, key=lambda t: t.priority)
        
        if not task.subtasks:
            # Leaf task: Execute with robust error handling
            output = self._execute_task_with_retries(task)
            
            # Check if execution succeeded
            if not task.failed:
                task.completed = True
                self.completed_task_ids.append(task.id)
                self.last_progress_step = self.step_count  # Mark progress
                
                # Check if we need to remove completed tasks
                if all(t.completed or t.failed for t in self.task_queue):
                    self.task_queue = [t for t in self.task_queue if not t.completed and not t.failed]
                    
                # Record in memory with timestamp
                memory_item = {
                    "task": task.desc, 
                    "output": output, 
                    "timestamp": time.time(),
                    "success": True
                }
                self.memory.append(memory_item)
                self._update_semantic_cache(memory_item)
                
                # Update error history for drift detection (success = low error)
                self.error_history.append(0.1)
                
                return {"status": "executed", "task": task.desc, "output": output}
            else:
                # Execution failed after all retries
                memory_item = {
                    "task": task.desc, 
                    "output": output, 
                    "timestamp": time.time(),
                    "success": False,
                    "failed": True
                }
                self.memory.append(memory_item)
                self._update_semantic_cache(memory_item)
                
                # Record high error for drift detection
                self.error_history.append(0.8)
                
                return {"status": "failed", "task": task.desc, "output": output}
        else:
            # Non-leaf: Plan next subtask
            next_task = self._get_next_subtask(task)
            if next_task:
                # Record planning in memory
                self.memory.append({
                    "task": task.desc, 
                    "plan": next_task.desc, 
                    "timestamp": time.time()
                })
                return {"status": "planned", "task": task.desc, "next": next_task.desc}
            else:
                # All subtasks completed or blocked
                return {"status": "waiting", "task": task.desc, "output": "Waiting for dependencies."}

    def _get_next_subtask(self, task: TaskNode) -> Optional[TaskNode]:
        """Get the next executable subtask with highest priority."""
        executable_subtasks = [t for t in task.subtasks 
                              if not t.completed and t.can_execute(self.completed_task_ids)]
        
        if not executable_subtasks:
            return None
            
        return max(executable_subtasks, key=lambda t: t.priority)

    def _execute_task_with_retries(self, task: TaskNode) -> str:
        """Execute a task with retry logic and fallbacks."""
        max_retries = task.max_retries
        backoff_base = 1.5  # Exponential backoff base
        
        for attempt in range(max_retries):
            result = self._try_execute_with_fallbacks(task)
            
            # Check if execution succeeded and break early
            if not result.startswith("Failed:"):
                return result
                
            # If still failed, increment retry counter
            task.retry_count += 1
            
            # Apply exponential backoff between retries
            backoff_time = (backoff_base ** attempt) * 0.5  # in seconds
            time.sleep(backoff_time)
                
        # After all retries exhausted, mark task as failed
        task.failed = True
        return f"Failed after {max_retries} attempts: {task.desc}"
        
    def _try_execute_with_fallbacks(self, task: TaskNode) -> str:
        """Try to execute task with primary tool, then fallbacks."""
        words = task.desc.split()
        tool_key = words[0].lower() if words else "default"
        context = self._get_enhanced_context(task)
        
        # Get primary tool
        primary_tool = self.tools.get(tool_key, lambda x: f"{self.personality}: Done - {x}")
        
        try:
            # First try with primary tool
            result = primary_tool(f"{context}{task.desc}")
            return result
        except Exception as e:
            primary_error = str(e)
            
            # Try fallback for this specific tool if available and not default
            if tool_key in self.fallback_tools and tool_key != "default":
                try:
                    fallback = self.fallback_tools[tool_key]
                    result = fallback(f"{context}{task.desc}")
                    return result
                except Exception:
                    # Specific fallback failed, don't try default
                    pass
                    
            # Return detailed error for better debugging
            return f"Failed: {task.desc} - {primary_error}"
            
    def _create_alternative_path(self, failed_task: TaskNode) -> TaskNode:
        """Create an alternative task to achieve the same goal as a failed task."""
        # Create a new description indicating this is an alternative approach
        new_desc = f"Try alternative approach: {failed_task.desc}"
        
        # Create a new task with slightly higher priority
        new_priority = min(1.0, failed_task.priority * 1.1)
        new_task = self.add_task(new_desc, new_priority)
        
        # Add semantic note about failure to avoid same approach
        failure_note = {
            "original_task": failed_task.desc,
            "failure_count": failed_task.retry_count,
            "timestamp": time.time()
        }
        
        new_task.metadata["alternative_to"] = failure_note
        
        return new_task

    def _replan_for_stagnation(self) -> Dict:
        """Handle stagnation by generating new approaches to stuck tasks."""
        # Find tasks that haven't made progress
        now = time.time()
        stuck_tasks = []
        
        for task in self.task_queue:
            if not task.completed and not task.failed:
                # Calculate staleness
                age = now - task.created_at
                # Tasks older than 60 seconds without completion are candidates
                if age > 60:
                    stuck_tasks.append((task, age))
                    
        if not stuck_tasks:
            return {"status": "healthy", "output": "No stagnation detected."}
            
        # Sort by age, oldest first
        stuck_tasks.sort(key=lambda x: x[1], reverse=True)
        oldest_task, age = stuck_tasks[0]
        
        # Create a new approach for the oldest stuck task
        new_desc = f"Alternative approach for: {oldest_task.desc}"
        new_task = self.add_task(new_desc, oldest_task.priority * 1.2)
        
        # Record the relationship between tasks
        new_task.metadata["replaces_task"] = oldest_task.id
        
        # Mark the old task as completed to prevent it from blocking
        oldest_task.completed = True
        self.completed_task_ids.append(oldest_task.id)
        
        return {
            "status": "replanned", 
            "output": f"Created new approach for stagnant task: {oldest_task.desc}",
            "new_task": new_task.desc
        }

    def _validate_tasks(self) -> int:
        """Actively validate tasks and remove irrelevant or stagnant ones."""
        now = time.time()
        tasks_removed = 0
        to_remove = []
        
        # First, identify tasks to remove and update priorities
        for i, task in enumerate(self.task_queue):
            if task.completed or task.failed:
                continue
                
            age = now - task.created_at
            
            # More aggressive decay for non-time-sensitive tasks
            if not task.is_time_sensitive:
                # Exponential decay based on age
                age_in_minutes = age / 60
                
                # Progressively decrease priority more as tasks age
                if age_in_minutes > 2:  # After 2 minutes
                    decay_factor = math.exp(-0.2 * age_in_minutes)
                    task.priority = max(0.1, task.priority * decay_factor)
            
            # For very old tasks, consider removing them entirely
            if age > self.max_task_age and not task.is_time_sensitive:
                low_relevance = task.priority < 0.2
                low_mentions = self._count_recent_mentions(task.desc) == 0
                
                if low_relevance and low_mentions:
                    # This task is old, low priority, and not mentioned recently
                    to_remove.append(i)
                    tasks_removed += 1
            # For low priority tasks, trigger redecomposition
            elif task.priority < 0.2 and task.subtasks:
                task.subtasks.clear()  # Clear existing subtasks
                self._decompose_task(task)  # Try fresh decomposition
        
        # Remove tasks marked for removal (in reverse order to maintain indices)
        for i in sorted(to_remove, reverse=True):
            del self.task_queue[i]
        
        # Second, check for redundant tasks and consolidate them
        task_groups = self._group_similar_tasks()
        for group in task_groups:
            if len(group) > 1:
                # Keep the highest priority task, remove others
                sorted_group = sorted(group, key=lambda t: t.priority, reverse=True)
                keep_task = sorted_group[0]
                
                for task in sorted_group[1:]:
                    if task in self.task_queue:  # Make sure it hasn't been removed already
                        self.task_queue.remove(task)
                        tasks_removed += 1
                    
                # Boost priority of the kept task
                keep_task.priority = min(1.0, keep_task.priority * 1.1)
        
        return tasks_removed
    
    def _count_recent_mentions(self, task_desc: str) -> int:
        """Count mentions of a task in recent memory."""
        recent_memory_cutoff = time.time() - (5 * 60)  # Last 5 minutes
        
        # Count mentions in memory
        count = 0
        for mem in self.memory:
            if not isinstance(mem, dict):
                continue
                
            timestamp = mem.get('timestamp', 0)
            if timestamp < recent_memory_cutoff:
                continue
                
            # Check if task is mentioned
            content = str(mem.get('task', '')) + str(mem.get('output', ''))
            if self._calculate_semantic_similarity(task_desc, content) > 0.3:
                count += 1
                
        return count
    
    def _group_similar_tasks(self) -> List[List[TaskNode]]:
        """Group similar tasks based on semantic similarity with caching."""
        groups = []
        processed_ids = set()
        sim_cache = {}  # Cache similarity calculations
        
        # Filter for active tasks only
        active_tasks = [t for t in self.task_queue if not t.completed and not t.failed]
        
        for i, task in enumerate(active_tasks):
            if task.id in processed_ids:
                continue
                
            # Start a new group with this task
            current_group = [task]
            processed_ids.add(task.id)
            
            # Find similar tasks
            for j, other_task in enumerate(active_tasks[i+1:]):
                if other_task.id not in processed_ids:
                    # Check cache first
                    cache_key = f"{task.id}-{other_task.id}"
                    
                    if cache_key not in sim_cache:
                        # Calculate similarity only if not in cache
                        sim_cache[cache_key] = self._calculate_semantic_similarity(
                            task.desc, other_task.desc
                        )
                        
                    # Lower threshold to 0.5 to catch more duplicates
                    if sim_cache[cache_key] > 0.5:
                        current_group.append(other_task)
                        processed_ids.add(other_task.id)
            
            # Only add groups with multiple tasks
            if len(current_group) > 1:
                groups.append(current_group)
                
        return groups

    def reflect(self) -> Dict:
        """Enhanced slow loop: Adjust priorities, validate tasks, and learn from memory."""
        if time.time() - self.last_reflection < 10:  # Every 10s
            return {"status": "skipped", "reason": "Too soon"}
        self.last_reflection = time.time()

        # Update priorities based on recency, temporal relevance, and success probability
        now = time.time()
        tasks_updated = 0
        
        # Check for tasks that might be stuck in partial completion
        partially_completed_tasks = []
        
        for task in self.task_queue:
            if task.completed or task.failed:
                continue
                
            # Apply temporal attention - boost time-sensitive tasks
            if task.is_time_sensitive:
                task.priority = min(1.0, task.priority + 0.05)
                tasks_updated += 1
            
            # Apply age-based adjustment with more aggressive decay
            age = now - task.created_at
            age_factor = math.exp(-0.03 * age / 60)  # Exponential decay with time (minutes)
            
            # Check for mentions in recent memory with semantic matching
            mentions = sum(1 for m in self.memory 
                          if isinstance(m, dict) and 
                          self._calculate_semantic_similarity(task.desc, str(m.get('task', ''))) > 0.3)
            
            # Increase priority based on mentions but scale by age factor
            old_priority = task.priority
            task.priority = min(1.0, task.priority + mentions * 0.05 * age_factor)
            
            # Check for partial completion
            if task.subtasks and any(s.completed for s in task.subtasks) and not all(s.completed for s in task.subtasks):
                subtasks_completed = sum(1 for s in task.subtasks if s.completed)
                subtasks_total = len(task.subtasks)
                completion_ratio = subtasks_completed / subtasks_total
                
                # If more than 50% complete, boost priority to finish it
                if completion_ratio > 0.5:
                    task.priority = min(1.0, task.priority + 0.1)
                    partially_completed_tasks.append(task)
            
            if old_priority != task.priority:
                tasks_updated += 1
                
            # Update subtasks recursively with more aggressive priority adjustments
            for subtask in task.subtasks:
                if not subtask.completed and not subtask.failed:
                    self._update_subtask_priority_aggressive(subtask, now)
        
        # Fix for partially completed tasks - boost remaining subtasks
        for task in partially_completed_tasks:
            for subtask in task.subtasks:
                if not subtask.completed and not subtask.failed:
                    subtask.priority = min(1.0, subtask.priority + 0.15)
        
        # Check for concept drift with enhanced detection
        drift_detected, drift_score = self._detect_concept_drift_enhanced()
        
        # Archive memory to history and update semantic cache
        if self.memory:
            self.history.extend(self.memory)
            self.memory.clear()
            
        return {
            "status": "reflected", 
            "tasks_updated": tasks_updated,
            "drift_detected": drift_detected,
            "drift_score": drift_score if drift_detected else 0
        }
    
    def _update_subtask_priority_aggressive(self, subtask: TaskNode, now: float) -> None:
        """Recursively update subtask priorities with more aggressive decay."""
        if subtask.completed or subtask.failed:
            return
            
        # Apply temporal weighting
        if subtask.is_time_sensitive:
            subtask.priority = min(1.0, subtask.priority + 0.05)
            
        # Age-based adjustment with faster decay
        age = now - subtask.created_at
        age_factor = math.exp(-0.02 * age / 60)  # Faster decay (minutes)
        
        # Check for mentions in recent memory with semantic matching
        mentions = sum(1 for m in self.memory 
                      if isinstance(m, dict) and 
                      self._calculate_semantic_similarity(subtask.desc, str(m.get('task', ''))) > 0.3)
        
        # Adjust priority with more weight to recency
        subtask.priority = min(1.0, subtask.priority + mentions * 0.05 * age_factor)
        
        # More aggressive decay for older subtasks
        if age > 120:  # 2 minutes
            subtask.priority = max(0.1, subtask.priority * 0.9)
        
        # Update child subtasks
        for child in subtask.subtasks:
            if not child.completed and not child.failed:
                self._update_subtask_priority_aggressive(child, now)

    def _get_context(self) -> str:
        """Build context from recent memory with temporal attention."""
        if not self.memory:
            return ""
            
        # Apply temporal weighting to recent memories
        now = time.time()
        weighted_memories = []
        
        for m in self.memory:
            if not isinstance(m, dict):
                continue
                
            timestamp = m.get('timestamp', now)
            age = (now - timestamp) / 60  # Age in minutes
            
            # Calculate temporal weight with decay factor
            weight = math.exp(-0.1 * (1 - self.temporal_decay_factor) * age)
            
            # Boost weight for time-sensitive items
            if any(indicator in str(m) for indicator in ["urgent", "immediately", "soon"]):
                weight *= 1.5
                
            weighted_memories.append((m, weight))
            
        # Sort by weight and take top 3
        weighted_memories.sort(key=lambda x: x[1], reverse=True)
        top_memories = weighted_memories[:3]
        
        # Extract content from top memories
        context_items = []
        for m, _ in top_memories:
            if 'output' in m:
                context_items.append(m['output'])
            elif 'task' in m:
                context_items.append(m['task'])
                
        return f"Context: {' | '.join(context_items)} | " if context_items else ""

    def _extract_core_concepts(self, text: str) -> List[str]:
        """Extract core concepts from text for better semantic understanding."""
        # Simple keyword extraction - split and clean
        words = text.lower().split()
        # Remove common stop words
        stop_words = ["the", "a", "an", "in", "on", "at", "to", "for", "by", "with", "about"]
        concepts = [w for w in words if w not in stop_words and len(w) > 2]
        # Return unique concepts
        return list(set(concepts))
        
    def _calculate_semantic_similarity(self, text1: str, text2: str) -> float:
        """Calculate semantic similarity with improved matching and length penalty."""
        # Extract concepts
        concepts1 = set(self._extract_core_concepts(text1))
        concepts2 = set(self._extract_core_concepts(text2))
        
        # Calculate Jaccard similarity with length penalty
        if not concepts1 or not concepts2:
            return 0.0
            
        # Calculate overlap
        intersection = concepts1.intersection(concepts2)
        union = concepts1.union(concepts2)
        overlap = len(intersection) / len(union)
        
        # Add length penalty (penalize texts with very different lengths)
        len1, len2 = len(text1), len(text2)
        if len1 == 0 or len2 == 0:
            return overlap
            
        # Calculate length ratio difference (0 to 1)
        length_penalty = min(1.0, abs(len1 - len2) / max(len1, len2))
        
        # Apply penalty - reduce similarity by up to 20% for length mismatch
        return overlap * (1 - length_penalty * 0.2)

    def _update_semantic_cache(self, memory_item: Dict) -> None:
        """Update semantic cache with new memory item for better retrieval with size limit."""
        # Extract text from memory item
        text = ""
        if "task" in memory_item:
            text += memory_item["task"] + " "
        if "output" in memory_item:
            text += memory_item["output"] + " "
        text = text.strip()
        
        # Extract key concepts
        concepts = self._extract_core_concepts(text)
        
        # Store in semantic cache with timestamp
        timestamp = memory_item.get("timestamp", time.time())
        
        # Create simplified entry with concepts, original memory
        cache_entry = {
            "concepts": concepts,
            "memory": memory_item,
            "timestamp": timestamp
        }
        
        # Use timestamp as cache key
        cache_key = str(timestamp)
        self.semantic_cache[cache_key] = cache_entry
        
        # Cap cache size at 100 entries, removing oldest
        if len(self.semantic_cache) > 100:
            oldest_key = min(self.semantic_cache.keys(), key=lambda k: float(k))
            del self.semantic_cache[oldest_key]

    def _get_enhanced_context(self, current_task: TaskNode) -> str:
        """Build rich context from memory with semantic relevance."""
        if not self.memory and not self.semantic_cache:
            return ""
            
        # First, try to find semantically relevant memories
        task_text = current_task.desc
        relevant_memories = self._find_relevant_memories(task_text)
        
        # Also consider parent task context if available
        if current_task.parent:
            parent_memories = self._find_relevant_memories(current_task.parent.desc)
            # Combine without duplicates
            for mem in parent_memories:
                if mem not in relevant_memories:
                    relevant_memories.append(mem)
        
        # If we don't have enough relevant memories, add some recent ones
        if len(relevant_memories) < 2:
            # Apply temporal weighting to recent memories
            now = time.time()
            weighted_memories = []
            
            for m in self.memory:
                if not isinstance(m, dict):
                    continue
                    
                timestamp = m.get('timestamp', now)
                age = (now - timestamp) / 60  # Age in minutes
                
                # Calculate temporal weight with decay factor
                weight = math.exp(-0.1 * (1 - self.temporal_decay_factor) * age)
                
                # Boost weight for time-sensitive items
                if any(indicator in str(m) for indicator in ["urgent", "immediately", "soon"]):
                    weight *= 1.5
                    
                weighted_memories.append((m, weight))
                
            # Sort by weight and take top ones
            weighted_memories.sort(key=lambda x: x[1], reverse=True)
            top_memories = weighted_memories[:3]
            
            # Add top temporal memories if not already in relevant memories
            for m, _ in top_memories:
                if m not in relevant_memories:
                    relevant_memories.append(m)
        
        # Create a context summary from the relevant memories
        context_parts = []
        
        for m in relevant_memories:
            if 'output' in m and m.get('success', True):  # Prefer successful outputs
                context_parts.append(f"Previously: {m['output']}")
            elif 'task' in m:
                if 'output' in m:
                    context_parts.append(f"Tried: {m['task']} → {m['output'][:50]}")
                else:
                    context_parts.append(f"Task: {m['task']}")
        
        # Create a more structured context
        if context_parts:
            # Add task-specific context
            task_path = self._get_task_path(current_task)
            if task_path:
                context_parts.insert(0, f"Working on: {task_path}")
                
            return f"Context: {' | '.join(context_parts)} | "
        
        return ""
    
    def _find_relevant_memories(self, query_text: str) -> List[Dict]:
        """Find semantically relevant memories based on a query."""
        query_concepts = self._extract_core_concepts(query_text)
        
        # If no meaningful concepts in query, return empty list
        if not query_concepts:
            return []
        
        # Find relevant memories from semantic cache
        relevant_items = []
        for key, entry in self.semantic_cache.items():
            # Calculate semantic relevance score
            memory_concepts = entry["concepts"]
            if not memory_concepts:
                continue
                
            # Calculate similarity
            sim_score = self._calculate_semantic_similarity(query_text, " ".join(memory_concepts))
            
            # If similarity is above threshold, add to relevant list
            if sim_score > self.semantic_threshold:
                relevant_items.append((entry["memory"], sim_score))
                
        # Sort by relevance score and get top items
        relevant_items.sort(key=lambda x: x[1], reverse=True)
        top_items = [item[0] for item in relevant_items[:3]]  # Get top 3 relevant memories
        
        return top_items
    
    def _get_task_path(self, task: TaskNode) -> str:
        """Get the hierarchical path to a task for better context."""
        path_parts = [task.desc]
        
        # Walk up the parent chain
        current = task.parent
        while current:
            path_parts.insert(0, current.desc)
            current = current.parent
            
        if len(path_parts) <= 1:
            return ""
            
        # Format as Main task > Subtask > Current task
        return " > ".join(path_parts)

    def _detect_concept_drift_enhanced(self) -> Tuple[bool, float]:
        """Enhanced concept drift detection with better metrics."""
        if len(self.error_history) < 5:
            return False, 0.0
            
        # Use recent errors only (last 10)
        recent_errors = self.error_history[-10:]
        
        # Calculate multiple statistics for better detection
        mean_error = sum(recent_errors) / len(recent_errors)
        variance = sum((e - mean_error) ** 2 for e in recent_errors) / len(recent_errors)
        std_dev = math.sqrt(variance)
        
        # Calculate trend (increasing errors)
        if len(recent_errors) >= 5:
            first_half = recent_errors[:len(recent_errors)//2]
            second_half = recent_errors[len(recent_errors)//2:]
            first_mean = sum(first_half) / len(first_half)
            second_mean = sum(second_half) / len(second_half)
            trend_factor = second_mean - first_mean
        else:
            trend_factor = 0
            
        # Combined drift score incorporates multiple factors
        drift_score = (std_dev * 0.5) + (mean_error * 0.3) + (trend_factor * 0.2)
        
        # Check if drift score exceeds threshold
        drift_detected = drift_score > self.drift_threshold
        
        if drift_detected:
            # Respond to drift with more sophisticated recovery
            self.error_history = []
            
            # Make ongoing tasks more exploratory (lower their certainty)
            for task in self.task_queue:
                if not task.completed and not task.failed:
                    task.success_probability *= 0.8  # Reduce confidence
                    
                    # For high drift, trigger explicit replanning of complex tasks
                    if drift_score > self.drift_threshold * 1.5 and task.subtasks:
                        # Recreate this task with a fresh decomposition
                        self._replan_task(task)
                    
        return drift_detected, drift_score
        
    def _replan_task(self, task: TaskNode) -> None:
        """Replan a task by updating its decomposition."""
        # Only replan if it has subtasks and isn't complete
        if not task.subtasks or task.completed or task.failed:
            return
            
        # Keep track of which subtasks were already completed
        completed_subtask_descs = [s.desc for s in task.subtasks if s.completed]
        
        # Clear existing subtasks and decompose again
        task.subtasks = []
        self._decompose_task(task)
        
        # Mark any new subtasks as completed if they match previously completed ones
        for subtask in task.subtasks:
            for completed_desc in completed_subtask_descs:
                if self._calculate_semantic_similarity(subtask.desc, completed_desc) > 0.7:
                    subtask.completed = True
                    break

    def run(self, goal: str, steps: int = 125, test_mode: bool = False) -> List[Dict]:
        """Run agent for N steps with optional test mode."""
        self.add_task(goal)
        results = []
        
        try:
            for step_num in range(steps):
                result = self.step()
                results.append(result)
                
                # Reflect periodically
                if self.step_count % 5 == 0:  # Reflect every 5 steps
                    reflection = self.reflect()
                    results.append(reflection)
                    
                    # If drift detected, potentially add a recovery task
                    if reflection.get('drift_detected', False):
                        self.add_task(f"Review progress on: {goal}", 0.8)
                
                # In test mode, inject test events to verify robustness
                if test_mode and step_num in self._get_test_trigger_points(steps):
                    test_event = self._inject_test_event(step_num, steps)
                    if test_event:
                        results.append(test_event)
                
                if not self.task_queue:  # Stop if done
                    break
                    
            return results
        except Exception as e:
            # Add error handling at the top level for robustness
            error_result = {
                "status": "error", 
                "error": str(e), 
                "step": self.step_count
            }
            results.append(error_result)
            return results
            
    def _get_test_trigger_points(self, total_steps: int) -> List[int]:
        """Get points during the run to inject test events."""
        # Test at 10%, 30%, 60%, and 80% of the way through
        return [
            int(total_steps * 0.1),
            int(total_steps * 0.3),
            int(total_steps * 0.6),
            int(total_steps * 0.8)
        ]
    
    def _inject_test_event(self, step_num: int, total_steps: int) -> Optional[Dict]:
        """Inject a test event to verify robustness."""
        # Early phase - test task dependency handling
        if step_num < total_steps * 0.2:
            return self._test_dependencies()
            
        # Mid phase - test error handling and recovery
        elif step_num < total_steps * 0.5:
            return self._test_error_handling()
            
        # Later phase - test drift detection
        elif step_num < total_steps * 0.8:
            return self._test_drift_detection()
            
        # Final phase - test long-term priority maintenance
        else:
            return self._test_priority_maintenance()
    
    def _test_dependencies(self) -> Dict:
        """Test dependency handling by creating interdependent tasks."""
        # Create two tasks with dependencies between them
        task1 = self.add_task("Test dependency task 1", 0.7)
        task2 = self.add_task("Test dependency task 2", 0.7)
        
        # Make task2 depend on task1
        task2.add_dependency(task1.id)
        
        return {
            "status": "test_inject", 
            "test_type": "dependencies",
            "description": "Injected dependency test tasks"
        }
    
    def _test_error_handling(self) -> Dict:
        """Test error handling by injecting a task that will fail."""
        # Add a special task that will trigger an error
        error_task = self.add_task("error_test This task will fail deliberately", 0.9)
        
        # Add a special tool that raises an exception
        self.tools["error_test"] = lambda x: exec("raise ValueError('Test error')")
        
        return {
            "status": "test_inject", 
            "test_type": "error_handling",
            "description": "Injected error-generating task"
        }
    
    def _test_drift_detection(self) -> Dict:
        """Test drift detection by simulating concept drift."""
        # Inject a series of errors to trigger drift detection
        for _ in range(5):
            self.error_history.append(random.uniform(0.7, 1.0))
            
        return {
            "status": "test_inject", 
            "test_type": "drift_detection",
            "description": "Injected error pattern for drift detection"
        }
    
    def _test_priority_maintenance(self) -> Dict:
        """Test priority maintenance with long-running tasks."""
        # Add a mix of time-sensitive and normal tasks
        task1 = self.add_task("Urgent test task with deadline today", 0.5)
        task2 = self.add_task("Normal test task without time pressure", 0.5)
        
        # Artificially age the normal task
        task2.created_at = time.time() - 120  # 2 minutes old
        
        return {
            "status": "test_inject", 
            "test_type": "priority_maintenance",
            "description": "Injected tasks for priority testing"
        }

    def run_test_suite(self) -> Dict:
        """Run a comprehensive test suite to verify agent functionality."""
        test_results = {
            "decomposition": self._test_decomposition(),
            "execution": self._test_execution(),
            "memory": self._test_memory(),
            "priority": self._test_priority(),
            "dependencies": self._test_dependency_execution(),
            "drift_recovery": self._test_drift_recovery(),
            "execution_retries": self._test_execution_retries()
        }
        
        # Calculate overall score
        success_count = sum(1 for result in test_results.values() if result["success"])
        overall_score = success_count / len(test_results)
        
        return {
            "status": "test_complete",
            "results": test_results,
            "overall_score": overall_score
        }
        
    def _test_decomposition(self) -> Dict:
        """Test task decomposition capabilities."""
        test_tasks = [
            "Create a website with user authentication and database",
            "Plan a family vacation and book flights",
            "Research market trends for electric vehicles"
        ]
        
        decomposition_depths = []
        for task_desc in test_tasks:
            task = TaskNode(task_desc)
            self._decompose_task(task)
            
            # Measure decomposition depth
            max_depth = self._get_max_depth(task)
            decomposition_depths.append(max_depth)
        
        # Success if average depth is at least 2
        avg_depth = sum(decomposition_depths) / len(decomposition_depths)
        success = avg_depth >= 2
        
        return {
            "success": success,
            "average_depth": avg_depth,
            "details": f"Tested {len(test_tasks)} decomposition scenarios"
        }
    
    def _get_max_depth(self, task: TaskNode, current_depth: int = 0) -> int:
        """Get the maximum depth of a task tree."""
        if not task.subtasks:
            return current_depth
            
        subtask_depths = [self._get_max_depth(st, current_depth + 1) for st in task.subtasks]
        return max(subtask_depths) if subtask_depths else current_depth
    
    def _test_execution(self) -> Dict:
        """Test task execution with retries and fallbacks."""
        # Create special test tools
        intermittent_fail_count = [0]  # Use list for mutable state
        
        def intermittent_fail(x):
            intermittent_fail_count[0] += 1
            if intermittent_fail_count[0] % 2 == 1:
                raise ValueError("Intermittent test failure")
            return f"Success on retry: {x}"
        
        self.tools["test_intermittent"] = intermittent_fail
        
        # Register fallback
        self.fallback_tools["test_intermittent"] = lambda x: f"Fallback handled: {x}"
        
        # Create and execute test task
        task = TaskNode("test_intermittent This task fails on first try")
        result = self._execute_task_with_retries(task)
        
        # Check if retry or fallback worked
        success = "Success on retry" in result or "Fallback handled" in result
        
        return {
            "success": success,
            "result": result,
            "retries": task.retry_count
        }
    
    def _test_memory(self) -> Dict:
        """Test memory and context retrieval."""
        # Add some test memories
        self.memory.append({
            "task": "Test memory alpha", 
            "output": "Result alpha",
            "timestamp": time.time()
        })
        
        time.sleep(0.1)
        
        self.memory.append({
            "task": "Test memory beta", 
            "output": "Result beta",
            "timestamp": time.time()
        })
        
        # Update semantic cache
        for mem in self.memory:
            self._update_semantic_cache(mem)
        
        # Test retrieval with similar query
        task = TaskNode("Test retrieving information about alpha")
        context = self._get_enhanced_context(task)
        
        # Success if context contains relevant memory
        success = "alpha" in context
        
        return {
            "success": success,
            "context_length": len(context),
            "contains_relevant": "alpha" in context
        }
    
    def _test_priority(self) -> Dict:
        """Test priority adjustment over time."""
        # Create time-sensitive and normal tasks
        urgent_task = TaskNode("Urgent task with deadline today")
        urgent_task.is_time_sensitive = True
        
        normal_task = TaskNode("Normal task without time constraint")
        
        # Artificially age the tasks
        now = time.time()
        urgent_task.created_at = now - 60  # 1 minute old
        normal_task.created_at = now - 120  # 2 minutes old
        
        # Store initial priorities
        initial_urgent = urgent_task.priority
        initial_normal = normal_task.priority
        
        # Add to task queue
        self.task_queue = [urgent_task, normal_task]
        
        # Run validation
        self._validate_tasks()
        
        # Check results
        urgent_priority_maintained = urgent_task.priority >= initial_urgent
        normal_priority_decayed = normal_task.priority < initial_normal
        
        success = urgent_priority_maintained and normal_priority_decayed
        
        return {
            "success": success,
            "urgent_maintained": urgent_priority_maintained,
            "normal_decayed": normal_priority_decayed,
            "urgent_priority": urgent_task.priority,
            "normal_priority": normal_task.priority
        }
    
    def _test_dependency_execution(self) -> Dict:
        """Test handling of task dependencies."""
        # Clear existing tasks
        self.task_queue = []
        self.completed_task_ids = []
        
        # Create dependent tasks
        task1 = TaskNode("First dependency test task")
        task2 = TaskNode("Second dependency test task")
        
        # Make task2 depend on task1
        task2.add_dependency(task1.id)
        
        self.task_queue = [task1, task2]
        
        # Try to get executable tasks
        executable_tasks = [t for t in self.task_queue 
                           if not t.completed and t.can_execute(self.completed_task_ids)]
        
        # Should only get task1
        first_check = len(executable_tasks) == 1 and executable_tasks[0].id == task1.id
        
        # Complete task1 and check again
        task1.completed = True
        self.completed_task_ids.append(task1.id)
        
        executable_tasks = [t for t in self.task_queue 
                           if not t.completed and t.can_execute(self.completed_task_ids)]
        
        # Should now include task2
        second_check = len(executable_tasks) == 1 and executable_tasks[0].id == task2.id
        
        success = first_check and second_check
        
        return {
            "success": success,
            "first_check": first_check,
            "second_check": second_check
        }

    def _test_drift_recovery(self) -> Dict:
        """Test drift detection and recovery mechanism."""
        # Clear existing error history
        self.error_history = []
        
        # Inject a series of high errors to trigger drift
        for _ in range(10):
            self.error_history.append(1.0)
            
        # Count tasks before drift detection
        pre_task_count = len(self.task_queue)
        
        # Add a test task with subtasks to check replanning
        test_task = self.add_task("Test drift recovery task")
        test_task._decompose_task(test_task)
        initial_success_probability = test_task.success_probability
        
        # Run drift detection
        drift_detected, score = self._detect_concept_drift_enhanced()
        
        # Check if drift was detected and recovery actions taken
        success = drift_detected and score > 0
        probability_adjusted = test_task.success_probability < initial_success_probability
        
        return {
            "success": success and probability_adjusted,
            "drift_detected": drift_detected,
            "drift_score": score,
            "probability_adjusted": probability_adjusted,
            "details": "Tests drift detection and task confidence adjustment"
        }
    
    def _test_execution_retries(self) -> Dict:
        """Test execution retry logic with a failing task."""
        # Create a tool that always fails
        self.tools["fail_test"] = lambda x: exec("raise ValueError('Test failure')")
        
        # Create a test task
        task = TaskNode("fail_test Testing retry logic")
        
        # Execute with retry logic
        result = self._execute_task_with_retries(task)
        
        # Verify task was marked as failed after retries
        success = task.failed and task.retry_count == task.max_retries - 1
        
        return {
            "success": success,
            "retries": task.retry_count,
            "task_failed": task.failed,
            "result": result,
            "details": "Verified task is marked failed after all retries exhausted"
        }

# Example usage
tools = {
    "say": lambda x: x,
    "book": lambda x: f"Booked: {x.split('book ')[-1]}",
    "plan": lambda x: f"Planned: {x.split('plan ')[-1]}",
    "search": lambda x: f"Searched for: {x.split('search ')[-1]}",
    "remind": lambda x: f"Set reminder: {x.split('remind ')[-1]}"
}

# Create agent with personality and tools
agent = BioLiteAgent(tools, "Witty and helpful")

# Register fallbacks for robustness
agent.register_fallback("book", lambda x: f"Alternative booking: {x.split('book ')[-1]}")
agent.register_fallback("default", lambda x: f"I'll try a different approach: {x}")

# Optional: Run test suite before using the agent
if __name__ == "__main__":
    print("Running test suite...")
    test_results = agent.run_test_suite()
    print(f"Test suite overall score: {test_results['overall_score']:.2f}")
    
    print("\nRunning full agent with task...")
    results = agent.run("Plan a trip and book a flight", 125)
    
    # Display first 5 results
    print("\nFirst 5 steps:")
    for r in results[:5]:
        print(json.dumps(r, indent=2))
        
    # Display summary
    completed = sum(1 for r in results if r.get("status") == "executed")
    planned = sum(1 for r in results if r.get("status") == "planned")
    drift_detected = any(r.get("drift_detected", False) for r in results if r.get("status") == "reflected")
    
    print(f"\nSummary: {len(results)} steps, {completed} tasks executed, {planned} planning steps")
    print(f"Concept drift detected: {drift_detected}")
    
    # Test with a more complex goal to demonstrate decomposition
    print("\nRunning with complex goal...")
    complex_agent = BioLiteAgent(tools, "Efficient and thorough")
    complex_results = complex_agent.run(
        "Research vacation options, create an itinerary, and book transportation",
        50,
        test_mode=True  # Enable test mode to verify robustness
    )
    
    # Display task decomposition
    root_tasks = complex_agent.task_queue
    print("\nTask Decomposition:")
    
    def print_task_tree(task, indent=0):
        status = "✓" if task.completed else "✗" if task.failed else "⋯"
        print(f"{'  ' * indent}{status} {task.desc} (priority: {task.priority:.2f})")
        for subtask in task.subtasks:
            print_task_tree(subtask, indent + 1)
    
    for task in root_tasks:
        print_task_tree(task)
