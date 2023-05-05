#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# the csv files i sent yesterday are the main dataframes that will be the first two arguments of below function.
# the function will filter the dataframes and run the model and will create 3 dataframes to feed into D3


# In[26]:


from flask import *
import pandas as pd
import numpy as np
import ast
import os

app = Flask(__name__)

@app.route("/")
def index():
    return render_template("PassingNetwork_web.html")


####################################################################################
# this function should be writing 3 csv files in data folder and update the visual##
####################################################################################

@app.route('/flask-route', methods=['GET', 'POST'])
def route():
    
    print('function started')
    # reding data from data folder
    
    df_starting_w = pd.read_csv(os.path.join(os.path.dirname(__file__), "static", "data", "df_starting2.csv"))
    df_starting_pass_w = pd.read_csv(os.path.join(os.path.dirname(__file__), "static", "data", "df_starting_pass2.csv"))
    df_route_w = pd.read_csv(os.path.join(os.path.dirname(__file__), "static", "data", "player_route.csv"))
    
    season_team = df_route_w['Season/Team'].iloc[len(df_route_w)-1]
    
    #team = 'La Liga 2004/2005 Barcelona' #it works when manually assigned
    team = request.args.get('selected_value')
    if team == None:
        team = season_team
    
    print('Selected value:', team)
    
    #excluded = None #int(request.args.get('int_val'))
    excluded = request.args.get('row_id')
    
    if excluded != None:
        excluded = int(excluded)
    
    print("excluded: ", excluded)
    
    
    
    
    
    # model preperation and model run after this section until the end of the code where csv files are written
    
    # location columns read as strings, something like '[123, 456]' 
    #the following code block converts them into lists withut quotes so they look like [123, 456]
    
    df_starting_w = df_starting_w.copy()
    df_starting_w.loc[:, 'average_location'] = df_starting_w['average_location'].apply(lambda x: ast.literal_eval(x))
    
    df_starting_pass_w = df_starting_pass_w.copy()
    df_starting_pass_w.loc[:, 'average_location_x'] = df_starting_pass_w['average_location_x'].apply(lambda x: ast.literal_eval(x))
    df_starting_pass_w.loc[:, 'average_location_y'] = df_starting_pass_w['average_location_y'].apply(lambda x: ast.literal_eval(x))
    
    # filter the season/team
    
    filtered_df_starting = df_starting_w[df_starting_w['Season/Team'] == team]
    filtered_df_starting_pass = df_starting_pass_w[df_starting_pass_w['Season/Team'] == team]

    df_starting = filtered_df_starting.copy()
    df_starting_pass = filtered_df_starting_pass.copy()    
    
    # create state transition matrix
    
    pivot_table = df_starting_pass.pivot(index='row_id_x', columns='row_id_y', values='Pass Probability')
    state_tr_matrix = pivot_table.to_numpy()
    state_tr_matrix = np.nan_to_num(state_tr_matrix, nan=0)
    
    print(state_tr_matrix)
    
    # extra row for row id 11
    new_row = np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1])
    new_arr = np.zeros((state_tr_matrix.shape[0] + 1, state_tr_matrix.shape[1]))

    new_arr[:-1,:] = state_tr_matrix
    new_arr[-1,:] = new_row

    P = new_arr
        
    # Markov application

    # finding steady state matrix by power iteration
    Pm = P ** 50

    # normalizing the rows

    np.fill_diagonal(Pm, 0.0)
    last_row = Pm.shape[0] - 1
    last_col = Pm.shape[1] - 1

    Pm[last_row, :] = 0.0
    Pm[last_row, last_col] = 1.0

    row_sums = Pm.sum(axis=1)
    Pm = Pm / row_sums[:, np.newaxis]
    
    #### finding goalie id
    
    goalie = df_starting.loc[df_starting['is_goalie'] == 1]
    goalie_id = goalie['row_id'].iloc[0]
    
    ######excluding clicked node##########
    
    if excluded == None or excluded == goalie_id or excluded == 11:
        Pm = Pm
    
    else:
        
        Pm[excluded, :] = 0.0
        Pm[: , excluded] = 0.0
        Pm[excluded, goalie_id] = 1.0

        row_sums = Pm.sum(axis=1)

        Pm = Pm / row_sums[:, np.newaxis]


    # initiating the sequence from goalkeeper

    n = Pm.shape[0] # number of nodes
    pi = np.zeros(n)
    pi[goalie_id] = 1 # set the probability of first node (goalkeeper) to 1

    current_node = goalie_id # we need to add some sort of identifier for the goalkeeper in the dataframe "df_starting" in order to initiate the sequence
    route = [current_node]


    cnt = 0
    
    # function for setting priorities in case of player passing to himself, or same passing sequence repeats over and over

    def requirements(lst, val, df):
    
        def in_list(lst, val):
        
            if val in lst:
                return False
            else:
                return True
        
        def is_progressive(lst, val, df):
        
            lst_x = df.loc[df['row_id'] == lst[-1], 'average_location'].iloc[0][0]
            val_x = df.loc[df['row_id'] == val, 'average_location'].iloc[0][0]
        
            if lst_x > val_x:
                return False
        
            else:
                return True
        
        return in_list(lst, val) and is_progressive(lst, val, df)

    while current_node != 11 and cnt < 1580: # continue until reaching node 12
        cnt += 1
    
        pi = np.dot(pi, Pm)
        pi_sorted = np.sort(pi)[::-1]

        for i in range(len(pi_sorted)):
        
            ind = np.where(pi == pi_sorted[i])[0][0]
            next_node = ind

            if len(route) > 1:
        
                if requirements(route, ind, df_starting):
                    next_node = ind
                    break
            
            else:
                break      

        route.append(next_node)
        current_node = next_node
    
        n = Pm.shape[0]
        pi = np.zeros(n)
        pi[current_node] = 1
    
    # importance of the nodes
    ratio_dict = {}

    col_totals = np.sum(Pm, axis=0)
    col_totals

    for i in range(len(col_totals)):
        ratio_dict[i] = col_totals[i]

    
    # calculating the route
    
    nodes_route = set(route)

    if len(nodes_route) > 7:
  
        passer = Pm[route[4], :]
        shooter = Pm[:, -1]



        sorted_indices = np.argsort(passer)[::-1]

        non_matching_indices = [i for i in sorted_indices if i not in route[0:5]]
        first_three_indices = non_matching_indices[:3]
        values = [shooter[i] for i in first_three_indices]
        max_shoot = max(values)
        highest_shooter_index = np.where(shooter == max_shoot)

        route_adj = route.copy()[0:5]

        route_adj.append(highest_shooter_index[0][0])
        route_adj.append(route[-1])
        
    
    else:
        route_adj = route.copy()

    # creating route dataframe

    selected_rows = []

    for i in route_adj:
        selected_row = df_starting.loc[df_starting['row_id'] == i]
        selected_rows.append(selected_row)

    route_df = pd.concat(selected_rows)
    route_df['node_radius_ratio'] = route_df['row_id'].map(ratio_dict)

    min_value = route_df['node_radius_ratio'].min()
    max_value = route_df['node_radius_ratio'].max()

    # normalize the column values between 1 and 2
    route_df['node_radius_ratio'] = 1 + (route_df['node_radius_ratio'] - min_value) / (max_value - min_value)
    route_df.loc[route_df['row_id'] == 11, 'node_radius_ratio'] = 1

    #print(route_df)

    shifted_df = route_df.shift(-1)

    # concatenate the original DataFrame and the shifted DataFrame horizontally
    concatenated_df = pd.concat([route_df, shifted_df], axis=1)

    # rename the new columns
    concatenated_df.columns = ['Season/Team', 'player_id', 'season_starts', 'average_location', 'player_from', 'row_id', 'is_goalie', 'node_radius_ratio',
                          'Season/Team_2', 'player_id_2', 'season_starts_2', 'average_location_2', 'player_from_2', 'row_id_2', 'is_goalie_2', 'node_radius_ratio_2']
    
    # fill the NaN values in the last row with the first row values
    concatenated_df.iloc[-1, -8:] = concatenated_df.iloc[-1, :8].values
    
    def writenodes(df):
        ndf = df.copy()
        ndf['x_pos_abs'] = ndf['average_location'].apply(lambda x: x[0])
        ndf['y_pos_abs'] = ndf['average_location'].apply(lambda x: x[1])
        ndf['x_pos_perc'] = ndf['x_pos_abs'] / 120
        ndf['y_pos_perc'] = ndf['y_pos_abs'] / 80
        ndf['name'] = ndf['player_from']

        return ndf.to_csv(os.path.join(os.path.dirname(__file__), "static", "data", "player_nodes.csv"), index=False, encoding='utf-8-sig')


    def writelinks(df):
        ndf = df.copy()
    
        freq_max = ndf['Frequency'].max()
        freq_min = ndf['Frequency'].min()

        ndf['pass_completed'] = ((ndf['Frequency'] - freq_min) / (freq_max - freq_min)) * 4
        ndf['source'] = ndf['Passing']
        ndf['target'] = ndf['Receiving']
        ndf['source_x'] = ndf['average_location_x'].apply(lambda x: x[0]/120)
        ndf['source_y'] = ndf['average_location_x'].apply(lambda x: x[1]/80)
        ndf['target_x'] = ndf['average_location_y'].apply(lambda x: x[0]/120)
        ndf['target_y'] = ndf['average_location_y'].apply(lambda x: x[1]/80)


        return ndf.to_csv(os.path.join(os.path.dirname(__file__), "static", "data", "player_links.csv"), index=False, encoding='utf-8-sig')

    def writeroute(df):
        ndf = df.copy()
        ndf['source'] = ndf['player_from']
        ndf['name'] = ndf['player_from']
        ndf['target'] = ndf['player_from_2']
        ndf['source_x'] = ndf['average_location'].apply(lambda x: x[0]/120)
        ndf['source_y'] = ndf['average_location'].apply(lambda x: x[1]/80)
        ndf['target_x'] = ndf['average_location_2'].apply(lambda x: x[0]/120)
        ndf['target_y'] = ndf['average_location_2'].apply(lambda x: x[1]/80)

        return ndf.to_csv(os.path.join(os.path.dirname(__file__), "static", "data", "player_route.csv"), index=False, encoding='utf-8-sig')
    
    # write csv files in /static/data/
    writenodes(df_starting)
    writelinks(df_starting_pass)
    writeroute(concatenated_df)

    return 'succes' #render_template("PassingNetwork_web.html")

if __name__ == "__main__":
    app.run(debug=True)

