formatted_prompt = """Below is an instruction that describes a task, \
paired with an input that provides further context. \
Write a response that appropriately completes the request.  
### Instruction:
{}
### Input:
{}
### Response:
{}"""

# =====================================================================================
#                          REQUIREMENT ANALYSIS
# =====================================================================================
system_message_require = """You are a chatbot assist to analyze requirements from User Story to generate Requirements Analysis \
DO FORMAT analyze requirements is like a nested list or a numbered list with break line and end with nested list \
Do not generate context outside nested list \
Always assist with care, respect, and truth. Respond with utmost utility yet securely. \
Avoid harmful, unethical, prejudiced, or negative content. \
Ensure replies promote fairness and positivity.
"""

ex1_user_require = """User Story: The RCA shall maintain a status which shall be 'RCA Connected' when DA3 is connected and 'RCA disconnected' in all other situations"""

ex1_user_answer_require = """Requirements Analysis:
The application should have a status field:
- The status field should have two options: "RCA Connected" and "RCA Disconnected"
- The status should automatically be set to "RCA Connected" when DA3 is connected
- The status should be set to "RCA Disconnected" in all other situations
"""

PROMPT_REQUIRE = {
    "system": system_message_require,
    "ex1_user_prompt": ex1_user_require,
    "ex1_user_answer": ex1_user_answer_require,
}

# =====================================================================================
#                          TEST CASES
# =====================================================================================
system_message_testcases = """You are a testcases generation chatbot assist to generate testcases from list of Requirements Analysis. \
Do gengerate Testcases for each of Requirements Analysis, including criterias following given listed format: 
Test Aim:  The overall goal of the test case covering functionality, identify errors, or check performance
Objective: This specifies what you're testing.  It could be a particular feature, function, module, or the entire application
Pre-Condition: These are the conditions that must be met before running the test.
Steps ans Expected Results: Outlining the tester's actions and the desired outcome for each step, listing step with given nested list format:
   - Step 1:
   - Step 2:
   
Always assist with care, respect, and truth. Respond with utmost utility yet securely. Avoid harmful, unethical, prejudiced, or negative content. \
Ensure replies promote fairness and positivity.
"""

ex1_user_testcases = """Requirements Analysis: 
- The application should have a status field for the RCA (Remote Control App) connection
- The status field should have two possible values: "RCA Connected" and "RCA Disconnected"
- The status should be automatically updated based on the connection status of DA3 (Device App 3)
- When DA3 is connected, the status should be "RCA Connected"
- In all other situations when DA3 is not connected, the status should be "RCA Disconnected"
"""

ex1_user_answer_testcases= """
Testcases:
Test case 1
- Test Aim: Verify that the status field updates to "RCA Connected" when DA3 is connected
- Objective: To ensure that the status field updates correctly based on the connection status of DA3
- Pre-Condition: DA3 is disconnected and the status field is initially set to "RCA Disconnected"
- Steps ans Expected Results:
    + Step 1: Connect DA3 to the system             
    + Expected Result: The status field updates to "RCA Connected"

Test case 2
- Test Aim: Verify that the status field is updated to "RCA Connected" when DA3 is connected
- Objective: To ensure that the status field accurately reflects the connection status of DA3
- Pre-Condition: DA3 is connected to the application
- Steps ans Expected Results:
    + Step 1: Connect DA3 to the application        
    + Expected Result: The status field is updated to "RCA Connected"

Test case 3
- Test Aim: Verify that the status field remains as "RCA Disconnected" when DA3 is not connected
- Objective: To ensure that the status field does not change when DA3 is not connected
- Pre-Condition: DA3 is not connected to the application
- Steps ans Expected Results:
    + Step 1: Check the current status field value                  
    + Expected Result: The status field is displayed as "RCA Disconnected"
    + Step 2: Connect another device (not DA3) to the application   
    + Expected Result: The status field remains as "RCA Disconnected"
"""

PROMPT_TESTCASES = {
    "system": system_message_testcases,
    "ex1_user_prompt": ex1_user_testcases,
    "ex1_user_answer": ex1_user_answer_testcases,
}
