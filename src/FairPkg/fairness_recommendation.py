import logging
# ===========================================================================================
# IDENTIFICATION OF FAIRNESS METRIC USING FAIRNESS TREE
# ===========================================================================================
def fairness_tree_metric(ftree_df):
    """Fairness metric recommendation based on the questionnaire"""
    # 
    # INPUT
    # ftree_df (Dataframe)     : Fairness tree questionnaire (fetch from csv file)
    # List of attributes in fairness tree questionnaire csv file
    # [1] "Node"            - Current node
    # [2] "Answer"          - Possible user response
    # [3] "Emphasizes       - Emphasis (remove)
    # [4] "Previous"        - Number of the previous node
    # [5] "Next"            - Number of the next node
    # [6] "Responses"       - 
    # [7] "Question"        - User question
    # [8] "Example"         -             
    # [9] "Recommendation"   
    #
    # OUTPUT
    # results (dictionary)        - Fairness metric recommendation along with the node number
    # ---------------------------------------------------------------------------------------
    results =  {}
    NODE_START=1 # Number of the node from which the questionnaire will start
    NODE_END=-1 # Number of the node at which the questionnaire will end
    node_=NODE_START # current node of the iteration
    while(node_ != NODE_END):
        # Step 1: Filter Node rows
        node_data=ftree_df[ftree_df['Node'] == node_] 
        node_data.reset_index(drop=True, inplace=True) # reset index
        # Break if it's the last node
        if(node_data.loc[0,'Next'] == NODE_END):
            break
        # Step 2: Question, Example, and Responses
        question_=node_data.loc[0,'Question'] # picking the questionnaire
        example_=node_data.loc[0,'Example']
        responses_=node_data.loc[0,'Responses']
        logging.info("QUESTION: {}".format(question_))
        if(len(example_) > 0):
            logging.info("EXAMPLE: {}".format(example_))
        logging.info("ANSWER: {}".format(responses_))
        if(len(responses_) == 0):
            responses_ = "Yes/No"
        # User response
        user_response_=input()
        logging.info("User response: {}".format(user_response_))
        if((user_response_.lower()=="y") or (user_response_.lower()=="yes")):
            user_response_ = "Y"
        elif((user_response_.lower()=="n") or (user_response_.lower()=="no")):
            user_response_="N"
        logging.info(user_response_)
        # Update node value
        node_data=node_data[node_data['Answer']==user_response_]
        node_data.reset_index(drop=True,inplace=True) # reset index
        node_=node_data.loc[0,'Next']
    results['node']=node_
    results['fairness_metric']=node_data.loc[0,'Recommendation']
    return results
# ===========================================================================================
# IDENTIFICATION OF MITIGATION METHOD
# ===========================================================================================
def mitigation_mapping_method(mitigation_mapping_info,fairness_metric_name):
    """Mitigation method based on the fairness metric"""
    #
    # INPUT
    # mitigation_mapping_info (DataFrame) : Mitigation methods information (fetch from csv file)
    # List of attributes in the mitigation methods csv file
    # [1] "Fairness Metric"     - Name of the fairness metric
    # [2] "Mitigation Method"   - Name of the mitigation method based on the fairness metric
    # [3] "Available"           - Mitigation method availability in the github repository
    #
    # OUTPUT
    # mitigation_methods (dictionary)   :   Mitigation methods
    # ---------------------------------------------------------------------------------------
    # filter rows based on the given fairness metric
    mitigation_methods_=mitigation_mapping_info[mitigation_mapping_info['Fairness Metric']==fairness_metric_name]
    # filter mitigation methods available in the EAI github repository
    mitigation_methods_=mitigation_methods_[mitigation_methods_['Available'] == True] 
    mitigation_methods_.reset_index(drop=True, inplace=True) # reset index
    # list of mitigation methods based on the input fairness metric
    mitigation_methods_=mitigation_methods_['Mitigation Method'].values.tolist()
    if(len(mitigation_methods_) > 1):
        logging.info("Mitigation methods recommended for {}".format(fairness_metric_name))
    for x, mitigation_method_ in enumerate(mitigation_methods_):
        logging.info('{} - {}'.format(x+1, mitigation_method_))
    logging.info("Select number between 1 - {}".format(len(mitigation_methods_)))
    user_response_ = input()
    user_response_ =int(user_response_)
    mitigation_methods_=mitigation_methods_[user_response_-1]
    return mitigation_methods_
# ===========================================================================================

        