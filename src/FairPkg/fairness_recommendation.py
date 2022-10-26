"""
IDENTIFICATION OF FAIRNESS METRIC USING FAIRNESS TREE
IDENTIFICATION OF MITIGATION METHOD
"""
import pandas as pd
import logging

logger = logging.getLogger(__name__)
logger.addHandler(logging.StreamHandler())
logger.setLevel(logging.INFO)


def fairness_tree_metric(ftree_df: pd.DataFrame) -> dict:
    """Fairness metric recommendation based on the questionnaire

    Args:
        ftree_df (pd.DataFrame): Fairness tree questionnaire (fetch from csv file).
            List of attributes in fairness tree questionnaire csv file
            1. "Node"            - Current node
            2. "Answer"          - Possible user response
            3. "Emphasizes       - Emphasis (remove)
            4. "Previous"        - Number of the previous node
            5. "Next"            - Number of the next node
            6. "Responses"       -
            7. "Question"        - User question
            8. "Example"         -
            9. "Recommendation"

    Returns:
        dict : Fairness metric recommendation along with the node number
    """

    results = {}
    NODE_START = 1  # Number of the node from which the questionnaire will start
    NODE_END = -1  # Number of the node at which the questionnaire will end
    current_node = NODE_START  # current node of the iteration

    while current_node != NODE_END:
        # Step 1: Filter Node rows
        node_data = ftree_df[ftree_df['Node'] == current_node]
        node_data.reset_index(drop=True, inplace=True)
        # Break if it's the last node
        if node_data.loc[0, 'Next'] == NODE_END:
            break

        # Step 2: Question, Example, and Responses
        question = node_data.loc[0, 'Question']
        example = node_data.loc[0, 'Example']
        possible_answers = node_data.loc[0, 'Responses']

        if pd.isna(example):
            example = ""

        if pd.isna(possible_answers):
            possible_answers = ""

        user_answer = get_user_answer(question, example, possible_answers)

        # Update node value
        node_data = node_data[node_data['Answer'].str.lower() == user_answer]
        node_data.reset_index(drop=True, inplace=True)
        current_node = node_data.loc[0, 'Next']

    results['node'] = current_node
    results['fairness_metric'] = node_data.loc[0, 'Recommendation']
    return results


def get_user_answer(question: str, example: str, possible_answers: str) -> str:
    """
    Ask user to insert its answer to a specific question.
    Args:
        question:
        example:
        possible_answers:
    Returns:
        str: User answer
    """
    print("QUESTION: {}".format(question))

    if example != "":
        print("EXAMPLE: {}".format(example))

    if possible_answers == "":
        possible_answers = "Yes/No"
    print("ANSWER: {}".format(possible_answers))

    # Get user answer
    while True:
        user_answer = input("Insert your answer:").strip().lower()
        logger.debug("User inserted the answer {}".format(user_answer))

        if user_answer in [answer.strip().lower() for answer in possible_answers.split('/')]:
            if user_answer == "yes":
                user_answer = "y"
            elif user_answer == "no":
                user_answer = "n"
            break
        else:
            print(f"Provided an invalid answer {user_answer}. Please insert of the following possible answers: "
                  f"{possible_answers}")
    logger.info(f"User answer is {user_answer}")

    return user_answer


def mitigation_mapping_method(mitigation_mapping_info: pd.DataFrame, fairness_metric_name: str) -> dict:
    """
    Mitigation method based on the fairness metric.

    Args:
        mitigation_mapping_info (pd.DataFrame) : Mitigation methods information (fetch from csv file)
            List of attributes in the mitigation methods csv file
                1. "Fairness Metric"     - Name of the fairness metric
                2. "Mitigation Method"   - Name of the mitigation method based on the fairness metric
                3. "Available"           - Mitigation method availability in the GitHub repository
        fairness_metric_name (str) : Name of the fairness metric

    Returns:
        dict : Mitigation methods
    """

    # filter rows based on the given fairness metric
    fair_mitigation_methods = mitigation_mapping_info[mitigation_mapping_info['Fairness Metric'] == fairness_metric_name]

    # filter mitigation methods available in the EAI GitHub repository
    fair_mitigation_methods = fair_mitigation_methods[fair_mitigation_methods['Available']]
    fair_mitigation_methods.reset_index(drop=True, inplace=True)

    # list of mitigation methods based on the input fairness metric
    list_mitigation_methods = fair_mitigation_methods['Mitigation Method'].values.tolist()

    if len(list_mitigation_methods) > 0:
        print("Mitigation methods recommended for {}".format(fairness_metric_name))

        for x, mitigation_method in enumerate(list_mitigation_methods):
            print('{} - {}'.format(x + 1, mitigation_method))

        # Get user answer
        while True:
            input_message = "Select number between 1-{}: ".format(len(list_mitigation_methods))
            user_response = input(input_message)
            try:
                user_response = int(user_response)
                if user_response not in range(1, len(list_mitigation_methods)+1):
                    print(f"{user_response} is out of range. Please try again.")
                else:
                    break
            except ValueError:
                print(f"{user_response} is not a valid integer. Please try again.")

        user_mitigation_method = list_mitigation_methods[user_response - 1]
    else:
        print("No Mitigation method available")
        user_mitigation_method = []

    logger.info(f"Selected mitigation method {user_mitigation_method}")

    return user_mitigation_method


if __name__ == "__main__":
    ftree_df = pd.read_csv("fairness_data/fairness_tree.csv")
    #results = fairness_tree_metric(ftree_df)

    mitigation_mapping_info = pd.read_csv("fairness_data/mitigation_mapping.csv")
    mitigation_method = mitigation_mapping_method(mitigation_mapping_info, 'Statistical Parity')
