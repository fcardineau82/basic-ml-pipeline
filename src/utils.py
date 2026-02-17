import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import pandas as pd

#function to match questions and answers
def match_questions_answers(df):
    """
    Matches questions with their corresponding answers based on IDs.

    Args:
        df (pd.DataFrame): DataFrame containing the parsed XML data.

    Returns:
        pd.DataFrame: DataFrame with matched questions and answers.
    """
    questions = df[df["PostTypeId"].astype(str) == "1"]
    answers = df[df["PostTypeId"].astype(str) == "2"]

    matched_data = []

    for _, question in questions.iterrows():
        question_id = question["Id"]
        related_answers = answers[answers["ParentId"] == question_id]
        
        for _, answer in related_answers.iterrows():
            matched_entry = {
                "QuestionId": question_id,
                "QuestionBody": question["Body"],
                "AnswerId": answer["Id"],
                "AnswerBody": answer["Body"]
            }
            matched_data.append(matched_entry)

    matched_df = pd.DataFrame(matched_data)
    return matched_df

#function to transfer the dataframe to csv
def dataframe_to_csv(df, file_path):
    df.to_csv(file_path, index=False)
    print(f"Dataframe saved to {file_path}")


#function to display the number of unanswered questions and how many questions have an answer accepted
def count_unanswered_and_accepted(df):
    """
    Counts unanswered questions and questions with accepted answers.

    Args:
        df (pd.DataFrame): DataFrame containing the parsed XML data.
    """
    questions = df[df["PostTypeId"].astype(str) == "1"]
    answers = df[df["PostTypeId"].astype(str) == "2"]

    # Count unanswered questions
    unanswered_count = len(questions[~questions["Id"].isin(answers["ParentId"].unique())])

    # Count questions with accepted answers
    accepted_count = len(questions[questions["AcceptedAnswerId"].notna()])

    # Count questions with any answers
    questions_with_answers = answers["ParentId"].unique()
    questions_with_answers_count = len(questions[questions["Id"].isin(questions_with_answers)])

    print(f"Number of unanswered questions: {unanswered_count}")
    print(f"Number of questions with accepted answers: {accepted_count}")
    print(f"Number of questions with at least one answer: {questions_with_answers_count}")

#function to draw two histogram , one for questions with score under the median score, one for questions with scores over. For both we remove outliers. 
def plot_question_score_histograms(df, body_length_limit=2000):
    """
    Plots histograms of question scores, separated by median score.

    Args:
        df (pd.DataFrame): DataFrame containing the parsed XML data.
    """
    # filter question into a new dataframe
    questions = df[df["PostTypeId"].astype(str) == "1"]
    #for each question check if the length of the body is under 2000 characters and add a new column to the dataframe with the length of the body recorded as an integer
    questions["BodyLength"] = questions["Body"].astype(str).apply(len)
    # filter questions with body length under 2000 characters
    short_questions = questions[questions["BodyLength"] < 2000]

    # create a dataframe with only the score column as integers and the corresponding body length
    scores = short_questions[["Score", "BodyLength"]].copy()
    scores["Score"] = scores["Score"].astype(int)
    
    # calculate the median score for that column
    median_score = scores.median()
    print(f"Median Score: {median_score}")
    # separate scores into low and high based on median
    low_scores = scores[scores["Score"] <= median_score["Score"]]
    high_scores = scores[scores["Score"] > median_score["Score"]]

    # Remove outliers using IQR and return cleaned data
    def remove_outliers(data):
        Q1 = data.quantile(0.25)
        Q3 = data.quantile(0.75)
        IQR = Q3 - Q1
        return data[(data >= (Q1 - 1.5 * IQR)) & (data <= (Q3 + 1.5 * IQR))]

    # Cleaned data into two new dataframes
    low_scores_clean = remove_outliers(low_scores)
    high_scores_clean = remove_outliers(high_scores)

    #plotting questions that have a lenght under 2000 characters on a histogram. separate overlaying histogram for high score and low score questions
    #create a new ax object that is a histogram of the low score questions
    ax = low_scores_clean.plot.hist(bins=60,density=True,  label='Low scores', color='blue', figsize=(16,10))
    #add the high score questions to the same histogram
    high_scores_clean.plot.hist(bins=60,density=True, alpha=0.7, label='High scores', color='green', ax=ax)
    #add title and labels
    plt.title('Histogram of Question Scores (Outliers Removed)')

    handles = [
        Rectangle((0,0),1,1,color=c, alpha=a) for c,a in [('blue',1.0), ('green',0.7)]
    ]
    labels= ['Low scores', 'High scores']   
    plt.legend(handles, labels)
    plt.xlabel('Question length (chars)')
    plt.ylabel('Density')
    plt.show()
   



   

  

