import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import precision_recall_fscore_support
from sklearn.model_selection import train_test_split
import argparse

def help():
    """Displays script usage information"""
    parser = argparse.ArgumentParser(
        description=(
            "Find out if you would have survived the Titanic catastrophe. Please note that this is purely fictional simulator that does not reflect reality and should\n be taken with a grain of salt :)\n\n"
            "Required Columns in csv file:\n"
            "  - Sex: Gender of the passenger (1 for male or 0 for female).\n"
            "  - Age: Age of the passenger in years.\n"
            "  - SibSp: Number of siblings or spouses aboard.\n"
            "  - Parch: Number of parents or children aboard.\n"
            "  - Fare: Ticket fare paid by the passenger.\n"
            "  - Pclass_1: 1st passenger class (1 if True, 0 if False).\n"
            "  - Pclass_2: 2nd passenger class (1 if True, 0 if False).\n"
            "  - Pclass_3: 3rd passenger class (1 if True, 0 if False).\n"
            "  - Embarked_C: You boarded in Cherbourg (1 if True, 0 if False).\n"
            "  - Embarked_Q: You boarded in Queens (1 if True, 0 if False).\n"
            "  - Embarked_S: You boarded in Southhampton (1 if True, 0 if False).\n"
            "  - Title_Millitary: You belong in the millitary social class (1 if True, 0 if False).\n"
            "  - Title_Higher_cls: You belong in the noble social class (1 if True, 0 if False).\n"
            "  - Title_Mob: Your belong in the commoner social class (1 if True, 0 if False).\n"
        ),
        formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument('-f', '--file', type=str, help='Path to your data file in CSV format. \nEnsure your file contains columns: Sex, Age, SibSp, Parch, Fare, Pclass_1, Pclass_2, Pclass_3, Embarked_C, Embarked_Q, Embarked_S, \nTitle_Higher_cls, Title_Millitary, Title_Mob.\n Use -h or --help option for explanation on what each column should contain. If not provided, the program will ask you for the input.')
    parser.add_argument('--stats', '-s',  action='store_true', help='Display evaluation statistics for the model')
    return parser

def preprocess(data):
  data = data.drop('PassengerId', axis='columns')
  data = data.drop('Ticket', axis='columns')
  data = data.drop('Cabin', axis='columns')

  # Changing the 'sex' column to binary values
  data["Sex"] = data["Sex"].apply(lambda x: 1 if x == 'male' else 0)

  # Replacing values ​​with corresponding columns
  data = pd.get_dummies(data, columns=["Pclass", "Embarked"], dtype=int)

  # Deleting rows with NaN values
  data = data.dropna()

  # Extracting information from the Name column
  data["Title"] = data["Name"].str.extract(r' ([A-Za-z]+)\.')

  title_replacements = {
      "Mlle": "Mob", "Ms": "Mob", "Mme": "Mob", "Mrs": "Mob", "Mr": "Mob", "Miss": "Mob", "Rev": "Higher_cls",
      "Lady": "Higher_cls", "Don": "Higher_cls",
      "Sir": "Higher_cls", "Capt": "Millitary", "Col": "Millitary", "Major": "Millitary", "Dr": "Higher_cls", "Master": "Higher_cls"
  } # 3 categories: Military, Higher class, Mob

  data["Title"] = data["Title"].replace(title_replacements)
  data = pd.get_dummies(data, columns=["Title"], dtype=int)

  # Removing the Name column after extracting information
  data = data.drop('Name', axis='columns')
  return data

def main():
    data = pd.read_csv("titanic.tsv", sep="\t")
    data = preprocess(data=data)
    features = [
        'Sex', 'Age', 'SibSp', 'Parch', 'Fare', 'Pclass_1', 'Pclass_2', 'Pclass_3', 'Embarked_C', 'Embarked_Q', 'Embarked_S', 'Title_Higher_cls', 'Title_Millitary', 'Title_Mob'
    ]

    parser = help()
    args = parser.parse_args()
    file_path = args.file

    # Division of data into training and test sets
    data_train, data_test = train_test_split(data, test_size=0.2, random_state=41)

    # Model training
    y_train = pd.Series(data_train["Survived"])
    x_train = pd.DataFrame(data_train[features])

    model = LogisticRegression(max_iter=500)  # Defining the model
    model.fit(x_train, y_train)  # Fitting the model

    # Predicting results for test data
    y_expected = pd.DataFrame(data_test["Survived"])
    x_test = pd.DataFrame(data_test[features])
    y_predicted = model.predict(x_test)  

    if args.stats:
        precision, recall, fscore, support = precision_recall_fscore_support(
            y_expected, y_predicted, average="binary", pos_label=1
        )

        print(f"Precision: {precision}")
        print(f"Recall: {recall}")
        print(f"F-score: {fscore}")

        score = model.score(x_test, y_expected)
        print(f"Model score: {score}")
    if args.file:
        user_data = pd.read_csv(file_path)
        x_user = user_data.reindex(columns=features, fill_value=0)
    else:
        sex = int(input("Enter your sex. 0 is for female, 1 is for male.: ").strip())
        age = float(input("Enter your age: ").strip())
        sibsp = int(input("Enter number of your siblings and/or spouses aboard: ").strip())
        parch = int(input("Enter number of your parents and/or children aboard: ").strip())
        fare = data['Fare'].mean()
        pclass = int(input("Enter your passenger class (1, 2, or 3): ").strip())
        if pclass == 1:
            Pclass_1 = 1
            Pclass_2 = 0
            Pclass_3 = 0
        elif pclass == 2:
            Pclass_1 = 0
            Pclass_2 = 1
            Pclass_3 = 0
        else:
            Pclass_1 = 0
            Pclass_2 = 0
            Pclass_3 = 1

        embarked = input("Enter port of embarkation (C, Q, or S), C = Cherbourg, Q = Queenstown, S = Southampton): ").strip().upper()
        if embarked == 'C':
            Embarked_C = 1
            Embarked_Q = 0
            Embarked_S = 0
        if embarked == 'Q':
            Embarked_C = 0
            Embarked_Q = 1
            Embarked_S = 0
        if embarked == 'S':
            Embarked_C = 0
            Embarked_Q = 0
            Embarked_S = 1

        social_class = input("Enter social class status (M, N, C), M = millitary, N = noble, C = commoner: ").strip().upper()
        if social_class == 'M':
            Title_Millitary = 1
            Title_Higher_cls = 0
            Title_Mob = 0
        elif social_class == 'N':
            Title_Millitary = 0
            Title_Higher_cls = 1
            Title_Mob = 0
        else:
            Title_Millitary = 0
            Title_Higher_cls = 0
            Title_Mob = 1

        user_data_dict = {
            "Sex": [sex],
            "Age": [age],
            "SibSp": [sibsp],
            "Parch": [parch],
            "Fare": [fare],
            "Pclass_1": [Pclass_1],
            "Pclass_2": [Pclass_2],
            "Pclass_3": [Pclass_3],
            "Embarked_C": [Embarked_C],
            "Embarked_Q": [Embarked_Q],
            "Embarked_S": [Embarked_S],
            "Title_Millitary": [Title_Millitary],
            "Title_Higher_cls": [Title_Higher_cls], 
            "Title_Mob": [Title_Mob]
        }

        user_data = pd.DataFrame.from_dict(user_data_dict)
        x_user = user_data.reindex(columns=features, fill_value=0)

    user_predictions = model.predict(x_user)
    if user_predictions == 1:
        answer = 'Congrats! You would totally survive.'
    else:
        answer = 'Yikes! Good thing you weren\'t on the Titanic! '
    print(f"Would you survive the Titanic Catastrophe? : {answer}")

if __name__ == "__main__":
    main()
