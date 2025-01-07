# %%
# Chain of Thought Agent Implementation
# This module implements a multi-agent system for processing complex tasks
# using chain-of-thought reasoning and specialized agent roles.

import json
import re
from groq_module3 import groq_llama3
import time

MODEL_NAME = "llama-3.3-70b-versatile"

class Context:
    """Context manager for maintaining state and memory across agent interactions"""
    def __init__(self):
        # Main state dictionary for current processing state
        self.state = {}
        # Short-term memory for recent interactions
        self.short_term_memory = []
        # Long-term memory for persistent information
        self.long_term_memory = []

    def update_state(self, key, value):
        # Updates the current state with new information
        self.state[key] = value

    def get_state(self, key):
        # Retrieves information from the current state
        return self.state.get(key, None)

    def add_to_short_term_memory(self, data):
        # Stores recent interactions in short-term memory
        self.short_term_memory.append(data)

    def add_to_long_term_memory(self, data):
        # Archives important information in long-term memory
        self.long_term_memory.append(data)

class EmbeddingModel:
    """Handles task analysis and breakdown using chain of thought reasoning"""
    def __init__(self, context):
        self.context = context

    def converse(self, user_input):
        # Process user input and generate structured task breakdown
        system_role = """
        Objective:
        You are an advanced embedding model responsible for analyzing user tasks and dividing them into steps using a Chain of Thought reasoning process. Each task must be divided into the following steps:

        - **Step 1:** Identify the foundational aspects, such as definitions, concepts, or introductory explanations.
            - **For coding-related tasks:** Provide a brief flow of the code, explaining the main components and structure.
        - **Step 2:** Build upon Step 1 by providing examples, applications, or real-world use cases.
            - **For coding-related tasks:** Check and list the necessary packages or libraries that will be required to implement the code.
        - **Step 3:** Offer deeper analysis, implications, or extended reasoning based on the outputs of Step 1 and Step 2.
            - **For coding-related tasks:** Write the specific functions or key parts of the code based on the general structure from Step 1 and the packages identified in Step 2.
        - **Step 4:** Use a function call to search for information from Google based on a suggested query. This includes finding and summarizing recent news articles or updates on the topic. Must include Reference Links at the end.
            - **For coding-related tasks:** Search for relevant coding functions, best practices, and additional libraries or packages that may help implement or optimize the code. Summarize and update based on this new information.

        Each step builds upon the previous ones to ensure a coherent and comprehensive understanding of the task.

        Output Requirements:
        You MUST always return the output in the following JSON format, starting with ```json on a new line, followed by the JSON object, and ending with ``` on a new line:

        ```json
        {
            "Step1": "Your output for Step 1",
            "Step2": "Your output for Step 2",
            "Step3": "Your output for Step 3",
            "Step4": "Your output for Step 4"
        }
        ```
        Do not include any additional text or explanation outside the JSON block.
        """
        embedding_output = groq_llama3(system_role, user_input, MODEL_NAME, functioncall=False)
        self.context.update_state('embedding_output', embedding_output)
        self.context.add_to_short_term_memory({'user_input': user_input, 'embedding_output': embedding_output})
        return embedding_output

    def extract_tasks(self, embedding_output):
        # Extract and parse individual tasks from the JSON formatted output
        try:
            json_pattern = r"```json\s*(\{[\s\S]*?\})\s*```"
            json_match = re.search(json_pattern, embedding_output)
            if json_match:
                raw_json = json_match.group(1)
                # Clean and parse the JSON content
                raw_json = raw_json.replace('\n', '').replace('\r', '').replace('\t', '')
                extracted_json = json.loads(raw_json)
                task1 = extracted_json.get("Step1", "No task assigned for Step 1")
                task2 = extracted_json.get("Step2", "No task assigned for Step 2")
                task3 = extracted_json.get("Step3", "No task assigned for Step 3")
                task4 = extracted_json.get("Step4", "No task assigned for Step 4")
                return task1, task2, task3, task4
            else:
                print("Embedding output did not contain expected JSON format.")
                print("Embedding Output:\n", embedding_output)
                raise ValueError("No JSON content found in the embedding output")
        except json.JSONDecodeError as e:
            print("JSONDecodeError:", e)
            print("Raw JSON string was:", raw_json)
            raise ValueError("Failed to decode JSON from the embedding output")

class Agent:
    """Specialized agent for executing specific tasks in the chain"""
    def __init__(self, context, name):
        self.context = context
        self.name = name

    def execute_task(self, task_input, previous_outputs=None):
        # Execute the assigned task, incorporating previous outputs if available
        system_role = f"You are {self.name}, specialised in executing a particular domain of tasks."
        if previous_outputs:
            system_role += f"\nYou should build upon the following information:\n{previous_outputs}"
        output = groq_llama3(system_role, task_input, MODEL_NAME, functioncall=False)
        self.context.add_to_short_term_memory({'agent': self.name, 'output': output})
        return output

class AgentValidator:
    """Validates and combines outputs from all agents"""
    def __init__(self, context):
        self.context = context

    def validate(self, combined_output, user_input):
        # Validate and enhance the combined output from all agents
        system_role = f"""
        You are responsible for validating and combining the outputs regarding user input: [{user_input}] from Steps 1 to 4 of the Chain of Thought process. Your task is to:
        1. **Validate** the outputs for correctness and consistency.
        2. **Combine** the steps into a coherent final answer.
        3. **Enhance** where needed for clarity.
        4. Decide if further iteration is needed to fully complete the task. If needed, set "next_action" to "iterate", otherwise set it to "stop".

        Each step represents:
        - **Step 1:** Definitions, concepts.
        - **Step 2:** Examples, applications.
        - **Step 3:** Deeper analysis, implications.
        - **Step 4:** Information from Internet and News. Include Reference Links.

        **For coding-related tasks**: If the input results are related to coding, your task is to combine them into one complete, runnable code. Ensure the code is logically structured and executable.

        Output Requirements:
        You MUST always return the output in the following JSON format, starting with ```json on a new line, followed by the JSON object, and ending with ``` on a new line:

        ```json
        {{
            "final_output": "Your combined and validated output here.",
            "next_action": "iterate" or "stop"
        }}
        ```
        Do not include any additional text or explanation outside the JSON block.
        """
        final_output_raw = groq_llama3(system_role, combined_output, MODEL_NAME, functioncall=False)
        
        # Parse and validate the output JSON
        try:
            json_pattern = r"```json\s*(\{[\s\S]*?\})\s*```"
            json_match = re.search(json_pattern, final_output_raw)
            if json_match:
                raw_json = json_match.group(1)
                raw_json = raw_json.replace('\n', '').replace('\r', '').replace('\t', '')
                extracted_json = json.loads(raw_json)
                final_output = extracted_json.get("final_output", "")
                next_action = extracted_json.get("next_action", "stop")
            else:
                raise ValueError("No JSON content found after ```json")
        except json.JSONDecodeError as e:
            print("JSONDecodeError:", e)
            print("Raw JSON string was:", raw_json)
            raise ValueError("Failed to decode JSON from the final output")
        
        # Store the validation results in long-term memory
        self.context.add_to_long_term_memory({'agent': 'agent5', 'output': final_output})
        return final_output, next_action

def main():
    # Initialize the context and agent system
    context = Context()

    # Create instances of all required agents
    embedding_model = EmbeddingModel(context)
    agent1 = Agent(context, 'agent1')
    agent2 = Agent(context, 'agent2')
    agent3 = Agent(context, 'agent3')
    agent4 = Agent(context, 'agent4')
    agent5 = AgentValidator(context)

    # Start the interaction loop
    user_input = input("User: ")

    # Initialize iteration control variables
    should_iterate = True
    iteration_count = 0
    max_iterations = 5  # Safety limit to prevent infinite loops

    while should_iterate and iteration_count < max_iterations:
        iteration_count += 1
        print(f"\n--- Iteration {iteration_count} ---\n")

        # Process the user input through the embedding model
        embedding_output = embedding_model.converse(user_input)
        print("Embedding Output:\n", embedding_output)

        # Extract individual tasks for each agent
        task1_input, task2_input, task3_input, task4_input = embedding_model.extract_tasks(embedding_output)

        # Execute Agent 1's task (Foundation)
        agent1_output = agent1.execute_task(task1_input)
        context.update_state('agent1_output', agent1_output)
        print(f"Agent1 Thinking...{task1_input}")
        time.sleep(0.5)

        # Execute Agent 2's task (Examples/Applications)
        agent2_output = agent2.execute_task(task2_input, previous_outputs=agent1_output)
        context.update_state('agent2_output', agent2_output)
        print(f"Agent2 Thinking...{task2_input}")
        time.sleep(0.5)

        # Execute Agent 3's task (Analysis)
        combined_previous_output = "Agent1 Output:\n" + agent1_output + "\nAgent2 Output:\n" + agent2_output
        agent3_output = agent3.execute_task(task3_input, previous_outputs=combined_previous_output)
        context.update_state('agent3_output', agent3_output)
        print(f"Agent3 Thinking...{task3_input}")
        time.sleep(0.5)

        # Execute Agent 4's task (External Information)
        combined_previous_output = (
            "Agent1 Output:\n" + agent1_output +
            "\nAgent2 Output:\n" + agent2_output +
            "\nAgent3 Output:\n" + agent3_output
        )
        agent4_output = agent4.execute_task(task4_input, previous_outputs=combined_previous_output)
        context.update_state('agent4_output', agent4_output)
        print(f"Agent4 Thinking...{task4_input}")
        time.sleep(0.5)

        # Combine all outputs for final validation
        combined_output = (
            "Step1: " + (agent1_output or '') + "\n" +
            "Step2: " + (agent2_output or '') + "\n" +
            "Step3: " + (agent3_output or '') + "\n" +
            "Step4: " + (agent4_output or '')
        )

        # Agent 5 validates and enhances the final output
        final_output, next_action = agent5.validate(combined_output, user_input)

        # Display the final results
        print("\nFinal Output from Agent 5:")
        print(final_output)
        print(f"\nAgent 5 suggests next action: {next_action}")

        # Handle iteration based on validation results
        if next_action.lower() == 'iterate':
            user_decision = input("Agent 5 suggests iterating again. Do you want to proceed? (y/n): ").strip().lower()
            if user_decision == 'y':
                # Get additional input for the next iteration
                user_input = input("Please provide additional input or press Enter to continue: ")
                if not user_input:
                    user_input = "Continue with the previous input."
                should_iterate = True
            else:
                should_iterate = False
        else:
            should_iterate = False

if __name__ == "__main__":
    main()

# %%
