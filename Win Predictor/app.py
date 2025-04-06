# Importing the necessary dependencies
import streamlit as st
import pandas as pd
import pickle
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder

# Declaring the teams
teams = ['Sunrisers Hyderabad',
         'Mumbai Indians',
         'Royal Challengers Bangalore',
         'Kolkata Knight Riders',
         'Kings XI Punjab',
         'Chennai Super Kings',
         'Rajasthan Royals',
         'Delhi Capitals']

# Declaring the venues
cities = ['Hyderabad', 'Bangalore', 'Mumbai', 'Indore', 'Kolkata', 'Delhi',
          'Chandigarh', 'Jaipur', 'Chennai', 'Cape Town', 'Port Elizabeth',
          'Durban', 'Centurion', 'East London', 'Johannesburg', 'Kimberley',
          'Bloemfontein', 'Ahmedabad', 'Cuttack', 'Nagpur', 'Dharamsala',
          'Visakhapatnam', 'Pune', 'Raipur', 'Ranchi', 'Abu Dhabi',
          'Sharjah', 'Mohali', 'Bengaluru']

# Load model
pipe = pickle.load(open('pipe.pkl', 'rb'))  # Ensure 'pipe.pkl' is in the same directory

# App title
st.title('🏏 IPL Win Predictor')

# Input form
col1, col2 = st.columns(2)

with col1:
    battingteam = st.selectbox('Select the Batting Team', sorted(teams))

with col2:
    bowlingteam = st.selectbox('Select the Bowling Team', sorted(teams))

# Validate team selection early
if battingteam == bowlingteam:
    st.error('Batting and Bowling teams must be different.')
    st.stop()

# Select city
city = st.selectbox('Select the Match City', sorted(cities))

# Target score input
target = int(st.number_input('Target Score', step=1))

# Game stats input
col3, col4, col5 = st.columns(3)

with col3:
    score = int(st.number_input('Current Score', step=1))

with col4:
    overs = float(st.number_input('Overs Completed', step=0.1))

with col5:
    wickets = int(st.number_input('Wickets Fallen', min_value=0, max_value=10, step=1))

# Match result scenarios (before prediction)
if score > target:
    st.success(f"{battingteam} has already won the match 🎉")
elif score == target - 1 and overs >= 20:
    st.info("Match Drawn 🤝")
elif wickets == 10 and score < target - 1:
    st.success(f"{bowlingteam} won the match 🏆")
elif wickets == 10 and score == target - 1:
    st.info("Match Tied 🟰")
else:
    if (0 <= target <= 300 and
        0 <= overs <= 20 and
        0 <= wickets <= 10 and
        score >= 0):

        try:
            if st.button('Predict Win Probability'):

                runs_left = target - score
                balls_left = int(120 - (overs * 6))
                wickets_left = 10 - wickets

                currentrunrate = score / overs if overs > 0 else 0
                requiredrunrate = (runs_left * 6) / balls_left if balls_left > 0 else 0

                # Input dataframe
                input_df = pd.DataFrame({
                    'batting_team': [battingteam],
                    'bowling_team': [bowlingteam],
                    'city': [city],
                    'runs_left': [runs_left],
                    'balls_left': [balls_left],
                    'wickets': [wickets],
                    'total_runs_x': [target],
                    'cur_run_rate': [currentrunrate],
                    'req_run_rate': [requiredrunrate]
                })

                # Prediction
                result = pipe.predict_proba(input_df)
                loss_prob = result[0][0]
                win_prob = result[0][1]

                # Display results
                st.subheader("🏁 Win Prediction")
                st.success(f"{battingteam}: {round(win_prob * 100)}% chance to win")
                st.error(f"{bowlingteam}: {round(loss_prob * 100)}% chance to win")

                # Optional: Add a progress bar for win probability
                st.progress(int(win_prob * 100))

        except ZeroDivisionError:
            st.error("Invalid input: overs or balls cannot be zero.")
    else:
        st.error("Please ensure all input values are within a valid range.")
