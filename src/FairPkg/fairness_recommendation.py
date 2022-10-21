"""
IDENTIFICATION OF FAIRNESS METRIC USING FAIRNESS TREE
"""
import pandas as pd
import logging

logger = logging.getLogger(__name__)


def fairness_tree_metric(ftree_df: pd.Dataframe) -> dict:
    """Fairness metric recommendation based on the questionnaire

    Args:
        ftree_df (pd.Dataframe): Fairness tree questionnaire (fetch from csv file).
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
        responses = node_data.loc[0, 'Responses']
        logging.info("QUESTION: {}".format(question))

        if len(example) > 0:
            logger.info("EXAMPLE: {}".format(example))

        if len(responses) == 0:
            responses = "Yes/No"
        logger.info("ANSWER: {}".format(responses))

        # User response
        user_response = input()
        logger.info("User response: {}".format(user_response))
        if (user_response.lower() == "y") or (user_response.lower() == "yes"):
            user_response = "Y"
        elif (user_response.lower() == "n") or (user_response.lower() == "no"):
            user_response = "N"
        logger.info(user_response)

        # Update node value
        node_data = node_data[node_data['Answer'] == user_response]
        node_data.reset_index(drop=True, inplace=True)
        current_node = node_data.loc[0, 'Next']

    results['node'] = current_node
    results['fairness_metric'] = node_data.loc[0, 'Recommendation']
    return results


# ===========================================================================================
# IDENTIFICATION OF MITIGATION METHOD
# ===========================================================================================

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
    fair_mitigation_methods = fair_mitigation_methods[fair_mitigation_methods['Available'] is True]
    fair_mitigation_methods.reset_index(drop=True, inplace=True)

    # list of mitigation methods based on the input fairness metric
    list_mitigation_methods = fair_mitigation_methods['Mitigation Method'].values.tolist()

    if len(list_mitigation_methods) > 1:
        logger.info("Mitigation methods recommended for {}".format(fairness_metric_name))
    for x, mitigation_method in enumerate(list_mitigation_methods):
        logger.info('{} - {}'.format(x + 1, mitigation_method))
    logger.info("Select number between 1 - {}".format(len(list_mitigation_methods)))
    user_response = input()
    try:
        user_response = int(user_response)
    except ValueError:
        print(f"{user_response} is not a valid integer. Please try again.")
    mitigation_methods = list_mitigation_methods[user_response - 1]
    return mitigation_methods
