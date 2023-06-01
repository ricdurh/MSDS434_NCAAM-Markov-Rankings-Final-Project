from flask import Flask, render_template_string
from flask import jsonify
import os
import random
from google.cloud import bigquery
import pandas as pd
import numpy as np
from scipy import linalg

import logging




app = Flask(__name__)
PORT = os.environ.get("PORT", 8080)
# Initialize BQ
client = bigquery.Client()



# Write SQL query to get the log model intercept and coeficient
query = """
    SELECT *
    FROM ML.WEIGHTS(MODEL `rich_dataset.log_reg_model`)
"""

# Run the query and load the result into dataframe
log_coef_df = client.query(query).to_dataframe()


# get the new data for this season
query_2 = """

        # get all games including neutral, but only if both teams from list above
        
        SELECT *
        FROM (
        
        SELECT h_market, a_market, prob as home_team_neutral_win_prob
        FROM ML.PREDICT( MODEL `rich_dataset.log_reg_model`,
        (

        WITH t1 as (
                SELECT season
                  , h_market
                  , a_market
                  , periods
                  , case when periods > 2.5 THEN 0 ELSE (h_points_game - a_points_game) END as margin_1

                FROM `bigquery-public-data.ncaa_basketball.mbb_games_sr`
                WHERE 1 = 1
                -- AND season IN(2017)
                AND h_points_game BETWEEN 1 AND 200
                AND h_conf_alias = a_conf_alias
                AND tournament_type IS NULL
        ),


        t2 as (
                SELECT season
                  , h_market
                  , a_market
                  , periods
                  , case when periods > 2.5 THEN 0 ELSE (a_points_game - h_points_game) END as margin_2

                FROM `bigquery-public-data.ncaa_basketball.mbb_games_sr`
                WHERE 1 = 1
                -- AND season IN(2017)
                AND h_points_game BETWEEN 1 AND 200
                AND h_conf_alias = a_conf_alias
                AND tournament_type IS NULL
        ), 

        t3 as (
                SELECT t1.season
                  , t1.h_market

                FROM t1 
                JOIN t2
                ON t1.h_market = t2.a_market
                AND t1.a_market = t2.h_market
                AND t1.season = t2.season

        ),


        t4 as (
                SELECT season
                , h_market
                , a_market
                , periods
                , neutral_site                         
                , case when periods > 2.5 THEN -1 ELSE (h_points_game - a_points_game) END as margin_1
                FROM `bigquery-public-data.ncaa_basketball.mbb_games_sr` 
                WHERE 1 = 1
                AND season IN(2017)
                AND ((tournament IS NULL) OR (tournament NOT IN ('NCAA')))
                AND h_market in (SELECT distinct h_market FROM t3 WHERE season in (2017))
                AND a_market in (SELECT distinct h_market FROM t3 WHERE season in (2017))
        )


        SELECT season
          , h_market
          , a_market
          , margin_1
          , case when neutral_site = FALSE THEN margin_1 ELSE margin_1 + 4 END as Margin
          , neutral_site
        FROM t4
        ORDER BY neutral_site
        
        ))

         , UNNEST(predicted_win_loss_label_probs) as probs
        WHERE probs.label = 1
        )


"""

new_teams_df = client.query(query_2).to_dataframe()

teams = new_teams_df['h_market'].unique()

num_teams = len(teams)

team_to_index = {team: idx for idx, team in enumerate(teams)}

pairwise_probs = np.zeros((num_teams, num_teams))


for _, row in new_teams_df.iterrows():
    home_team_idx = team_to_index[row['h_market']]
    away_team_idx = team_to_index[row['a_market']]
    prob = row['home_team_neutral_win_prob']
    pairwise_probs[home_team_idx, away_team_idx] = prob
    pairwise_probs[away_team_idx, home_team_idx] = 1 - prob

# Step 2: Construct the transition matrix
transition_matrix = pairwise_probs

# Step 3: Normalize rows of the transition matrix
row_sums = transition_matrix.sum(axis=1, keepdims=True)
transition_matrix /= row_sums

# Step 4: Compute the stationary distribution
eigvals, eigvecs = np.linalg.eig(transition_matrix.T)
stationary_distribution = np.real(eigvecs[:, np.isclose(eigvals, 1)].squeeze())
stationary_distribution /= stationary_distribution.sum()

# Create a dictionary mapping team names to their stationary distribution values
team_stationary_distribution = {team: stationary_distribution[idx] for team, idx in team_to_index.items()}

# Sort the dictionary by the stationary distribution values (lower values indicate better teams)
sorted_teams = sorted(team_stationary_distribution.items(), key=lambda x: x[1], reverse=False)

final_df = pd.DataFrame(sorted_teams[:30], columns=['Teams', 'Rank'])
final_df['Rank'] = [x + 1 for x in range(0,30)]


@app.route("/", methods=["GET"])
def hello():
    # adding logging statement
    logging.info('Processing request for /')
    html_table = final_df.to_html()
    return render_template_string("<html><body>{{ table | safe }}</body></html>", table=html_table)
    
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=PORT)
