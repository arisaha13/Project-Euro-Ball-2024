#load libraries
import pandas as pd
import numpy as np
import seaborn as sns
sns.set_theme(style='whitegrid', palette='pastel')
import matplotlib.pyplot as plt
from functools import reduce
import requests
from bs4 import BeautifulSoup
from urllib.parse import urljoin
import time

###obtaining  copa america 2019 match data
ca19_countries = ['Argentina', 'Bolivia', 'Brazil', 'Chile', 'Colombia', 'Ecuador', 'Japan', 'Paraguay', 'Peru', 'Qatar', 'Uruguay', 'Venezuela']

ca19_url = 'https://fbref.com/en/comps/685/2019/schedule/2019-Copa-America-Scores-and-Fixtures'
reqs = requests.get(ca19_url)
soup = BeautifulSoup(reqs.text, 'html.parser')

ca19_urls = []
for link in soup.find_all('a'):
    ca19_urls.append(link.get('href'))

ca19_urls_str = [str(i) for i in ca19_urls]
ca19_games = [s for s in ca19_urls_str if any(k in s for k in ca19_countries)]
ca19_games = [s for s in ca19_games if "2019-Copa-America" in s]
ca19_games = list(dict.fromkeys(ca19_games))
print(ca19_games)
len(ca19_games) == 26

ca19_games = ['https://fbref.com' + i for i in ca19_games]

ca19 = pd.DataFrame()
def ca19_match_data(link):

    teams = {}
    for i in range(0, len(ca19_countries)):
        if link.find(ca19_countries[i]) > 0:
            teams.update({ca19_countries[i]: link.find(ca19_countries[i])});
    #teams
    #print(min(teams, key=teams.get))

    team_stand = pd.read_html(link)[3]
    team_stand.columns = team_stand.columns.droplevel(0)
    team_stand = team_stand[['Gls', 'Ast', 'Sh', 'SoT', 'xG', 'npxG', 'SCA', 'GCA', 'PrgP', 'PrgC']]
    team_stand.columns = ['Goals', 'Assists', 'Shots', 'Shots_on_Target', 'xG', 'NPxG', 'SCA', 'GCA', 'Progressive_Passes', 'Progressive_Carries']
    team_stand = team_stand.rename(index={len(team_stand)-1: min(teams, key=teams.get)})
    team_stand = team_stand.iloc[-1:]
    #team_stand

    team_pass = pd.read_html(link)[4]
    team_pass.columns = team_pass.columns.droplevel(0)
    passing = ["Player", "#", "Pos", "Age", "Min", "passes_completed", "passes", "passes_pct", "passes_total_distance","passes_progressive_distance","passes_completed_short","passes_short","passes_pct_short","passes_completed_medium","passes_medium","passes_pct_medium","passes_completed_long","passes_long","passes_pct_long","assists","xa_chain","xa","key_passes","passes_into_final_third","passes_into_penalty_area","crosses_into_penalty_area","progressive_passes"]
    team_pass.columns = passing
    team_pass = team_pass.iloc[:, [9, 10, 12, 13, 15, 16, 18, 21, 25]]
    team_pass.columns = ['Progressive_Pass_Dist', 'Short_Passes_Completed', 'Short_Pass_%','Medium_Passes_Completed','Medium_Pass_%','Long_Passes_Completed','Long_Pass_%','xA', 'Crosses_into_Penalty_Area']
    team_pass = team_pass.rename(index={len(team_pass)-1: min(teams, key=teams.get)})
    team_pass = team_pass.iloc[-1:]
    #team_pass

    team_def = pd.read_html(link)[6]
    team_def.columns = team_def.columns.droplevel(0)
    team_def = team_def[['Tkl', 'TklW', 'Blocks', 'Int', 'Clr']]
    team_def.columns = ['Tackles', 'Challenges', 'Tackles_Won', 'Blocks', 'Interceptions', 'Clearances']
    team_def = team_def.drop(['Challenges'], axis=1)
    team_def = team_def.rename(index={len(team_def)-1: min(teams, key=teams.get)})
    team_def = team_def.iloc[-1:]
    #team_def

    team_pos = pd.read_html(link)[7]
    team_pos.columns = team_pos.columns.droplevel(0)
    team_pos = team_pos[['Touches', 'Def 3rd', 'Mid 3rd', 'Att 3rd', 'Succ%', '1/3']]
    team_pos.columns = ['Touches', 'Touches_Def_3rd', 'Touches_Mid_3rd', 'Touches_Att_3rd', 'Take_On_Success_%', 'Carries_into_Final_3rd']
    team_pos = team_pos.rename(index={len(team_pos)-1: min(teams, key=teams.get)})
    team_pos = team_pos.iloc[-1:]
    #team_pos

    team_misc = pd.read_html(link)[8]
    team_misc.columns = team_misc.columns.droplevel(0)
    team_misc = team_misc[['Recov', 'Won%']]
    team_misc.columns = ['Ball_Recoveries', 'Aerial_Duels_%']
    team_misc = team_misc.rename(index={len(team_misc)-1: min(teams, key=teams.get)})
    team_misc = team_misc.iloc[-1:]
    #team_misc

    team_gk = pd.read_html(link)[9]
    team_gk.columns = team_gk.columns.droplevel(0)
    team_gk = team_gk[['PSxG', '#OPA', 'AvgDist']]
    team_gk.columns = ['Post_Shot_xG', 'Def_Actions_Outside_Area', 'Avg_Dist_Actions_Outside_Area']
    team_gk = team_gk.rename(index={len(team_gk)-1: min(teams, key=teams.get)})
    team_gk = team_gk.iloc[-1:]
    #team_gk

    team_dfs = [team_stand, team_pass, team_def, team_pos, team_misc, team_gk]
    team_merge = reduce(lambda  left, right: pd.merge(left, right, left_index=True, right_index=True), team_dfs)
    team_merge = team_merge.apply(pd.to_numeric)
    #team_merge

    team2_stand = pd.read_html(link)[10]
    team2_stand.columns = team2_stand.columns.droplevel(0)
    team2_stand = team2_stand[['Gls', 'Ast', 'Sh', 'SoT', 'xG', 'npxG', 'SCA', 'GCA', 'PrgP', 'PrgC']]
    team2_stand.columns = ['Goals', 'Assists', 'Shots', 'Shots_on_Target', 'xG', 'NPxG', 'SCA', 'GCA', 'Progressive_Passes', 'Progressive_Carries']
    team2_stand = team2_stand.rename(index={len(team2_stand)-1: max(teams, key=teams.get)})
    team2_stand = team2_stand.iloc[-1:]
    #team2_stand

    team2_pass = pd.read_html(link)[11]
    team2_pass.columns = team2_pass.columns.droplevel(0)
    passing = ["Player", "#", "Pos", "Age", "Min", "passes_completed", "passes", "passes_pct", "passes_total_distance","passes_progressive_distance","passes_completed_short","passes_short","passes_pct_short","passes_completed_medium","passes_medium","passes_pct_medium","passes_completed_long","passes_long","passes_pct_long","assists","xa_chain","xa","key_passes","passes_into_final_third","passes_into_penalty_area","crosses_into_penalty_area","progressive_passes"]
    team2_pass.columns = passing
    team2_pass = team2_pass.iloc[:, [9, 10, 12, 13, 15, 16, 18, 21, 25]]
    team2_pass.columns = ['Progressive_Pass_Dist', 'Short_Passes_Completed', 'Short_Pass_%','Medium_Passes_Completed','Medium_Pass_%','Long_Passes_Completed','Long_Pass_%','xA', 'Crosses_into_Penalty_Area']
    team2_pass = team2_pass.rename(index={len(team2_pass)-1: max(teams, key=teams.get)})
    team2_pass = team2_pass.iloc[-1:]
    #team2_pass

    team2_def = pd.read_html(link)[13]
    team2_def.columns = team2_def.columns.droplevel(0)
    team2_def = team2_def[['Tkl', 'TklW', 'Blocks', 'Int', 'Clr']]
    team2_def.columns = ['Tackles', 'Challenges', 'Tackles_Won', 'Blocks', 'Interceptions', 'Clearances']
    team2_def = team2_def.drop(['Challenges'], axis=1)
    team2_def = team2_def.rename(index={len(team2_def)-1: max(teams, key=teams.get)})
    team2_def = team2_def.iloc[-1:]
    #team2_def

    team2_pos = pd.read_html(link)[14]
    team2_pos.columns = team2_pos.columns.droplevel(0)
    team2_pos = team2_pos[['Touches', 'Def 3rd', 'Mid 3rd', 'Att 3rd', 'Succ%', '1/3']]
    team2_pos.columns = ['Touches', 'Touches_Def_3rd', 'Touches_Mid_3rd', 'Touches_Att_3rd', 'Take_On_Success_%', 'Carries_into_Final_3rd']
    team2_pos = team2_pos.rename(index={len(team2_pos)-1: max(teams, key=teams.get)})
    team2_pos = team2_pos.iloc[-1:]
    #team2_pos

    team2_misc = pd.read_html(link)[15]
    team2_misc.columns = team2_misc.columns.droplevel(0)
    team2_misc = team2_misc[['Recov', 'Won%']]
    team2_misc.columns = ['Ball_Recoveries', 'Aerial_Duels_%']
    team2_misc = team2_misc.rename(index={len(team2_misc)-1: max(teams, key=teams.get)})
    team2_misc = team2_misc.iloc[-1:]
    #team2_misc

    team2_gk = pd.read_html(link)[16]
    team2_gk.columns = team2_gk.columns.droplevel(0)
    team2_gk = team2_gk[['PSxG', '#OPA', 'AvgDist']]
    team2_gk.columns = ['Post_Shot_xG', 'Def_Actions_Outside_Area', 'Avg_Dist_Actions_Outside_Area']
    team2_gk = team2_gk.rename(index={len(team2_gk)-1: max(teams, key=teams.get)})
    team2_gk = team2_gk.iloc[-1:]
    #team2_gk

    team2_dfs = [team2_stand, team2_pass, team2_def, team2_pos, team2_misc, team2_gk]
    team2_merge = reduce(lambda  left, right: pd.merge(left, right, left_index=True, right_index=True), team2_dfs)
    team2_merge = team2_merge.apply(pd.to_numeric)
    #team2_merge

    if (team_merge['Goals'][0] > team2_merge['Goals'][0]) == True:
        team_merge['Result'] = 3
        team2_merge['Result'] = 0
    elif (team_merge['Goals'][0] == team2_merge['Goals'][0]) == True:
        team_merge['Result'] = 1
        team2_merge['Result'] = 1
    elif (team_merge['Goals'][0] < team2_merge['Goals'][0]) == True:
        team_merge['Result'] = 0
        team2_merge['Result'] = 3;
    #team_merge
    #team2_merge

    df1_name = (min(teams, key=teams.get) + link.split(max(teams, key=teams.get), 1)[1]).replace("-", "_")
    team_merge = team_merge.rename(index={min(teams, key=teams.get): df1_name})
    #team_merge

    df2_name = (max(teams, key=teams.get) + link.split(max(teams, key=teams.get), 1)[1]).replace("-", "_")
    team2_merge = team2_merge.rename(index={max(teams, key=teams.get): df2_name})
    #team2_merge

    #ca19_game1 = pd.concat([team_merge, team2_merge], axis=0)
    time.sleep(60)
    global ca19
    ca19 = pd.concat([ca19, team_merge, team2_merge], axis=0)
    #return (team_merge, team2_merge)


for i in range(0, len(ca19_games)):
    ca19_match_data(ca19_games[i]);

ca19.to_csv("data_tables\copa19.csv", index=True)


###Obtaining copa america 2021 match data
ca21_countries = ['Argentina', 'Bolivia', 'Brazil', 'Chile', 'Colombia', 'Ecuador', 'Paraguay', 'Peru', 'Uruguay', 'Venezuela']

ca21_url = 'https://fbref.com/en/comps/685/2021/schedule/2021-Copa-America-Scores-and-Fixtures'
reqs = requests.get(ca21_url)
soup = BeautifulSoup(reqs.text, 'html.parser')

ca21_urls = []
for link in soup.find_all('a'):
    ca21_urls.append(link.get('href'))

ca21_urls_str = [str(i) for i in ca21_urls]
ca21_games = [s for s in ca21_urls_str if any(k in s for k in ca21_countries)]
ca21_games = [s for s in ca21_games if "2021-Copa-America" in s]
ca21_games = list(dict.fromkeys(ca21_games))
print(ca21_games)
len(ca21_games) == 28

ca21_games = ['https://fbref.com' + i for i in ca21_games]

ca21 = pd.DataFrame()
def ca21_match_data(link):

    teams = {}
    for i in range(0, len(ca21_countries)):
        if link.find(ca21_countries[i]) > 0:
            teams.update({ca21_countries[i]: link.find(ca21_countries[i])});
    #teams
    #print(min(teams, key=teams.get))

    team_stand = pd.read_html(link)[3]
    team_stand.columns = team_stand.columns.droplevel(0)
    team_stand = team_stand[['Gls', 'Ast', 'Sh', 'SoT', 'xG', 'npxG', 'SCA', 'GCA', 'PrgP', 'PrgC']]
    team_stand.columns = ['Goals', 'Assists', 'Shots', 'Shots_on_Target', 'xG', 'NPxG', 'SCA', 'GCA', 'Progressive_Passes', 'Progressive_Carries']
    team_stand = team_stand.rename(index={len(team_stand)-1: min(teams, key=teams.get)})
    team_stand = team_stand.iloc[-1:]
    #team_stand

    team_pass = pd.read_html(link)[4]
    team_pass.columns = team_pass.columns.droplevel(0)
    passing = ["Player", "#", "Pos", "Age", "Min", "passes_completed", "passes", "passes_pct", "passes_total_distance","passes_progressive_distance","passes_completed_short","passes_short","passes_pct_short","passes_completed_medium","passes_medium","passes_pct_medium","passes_completed_long","passes_long","passes_pct_long","assists","xa_chain","xa","key_passes","passes_into_final_third","passes_into_penalty_area","crosses_into_penalty_area","progressive_passes"]
    team_pass.columns = passing
    team_pass = team_pass.iloc[:, [9, 10, 12, 13, 15, 16, 18, 21, 25]]
    team_pass.columns = ['Progressive_Pass_Dist', 'Short_Passes_Completed', 'Short_Pass_%','Medium_Passes_Completed','Medium_Pass_%','Long_Passes_Completed','Long_Pass_%','xA', 'Crosses_into_Penalty_Area']
    team_pass = team_pass.rename(index={len(team_pass)-1: min(teams, key=teams.get)})
    team_pass = team_pass.iloc[-1:]
    #team_pass

    team_def = pd.read_html(link)[6]
    team_def.columns = team_def.columns.droplevel(0)
    team_def = team_def[['Tkl', 'TklW', 'Blocks', 'Int', 'Clr']]
    team_def.columns = ['Tackles', 'Challenges', 'Tackles_Won', 'Blocks', 'Interceptions', 'Clearances']
    team_def = team_def.drop(['Challenges'], axis=1)
    team_def = team_def.rename(index={len(team_def)-1: min(teams, key=teams.get)})
    team_def = team_def.iloc[-1:]
    #team_def

    team_pos = pd.read_html(link)[7]
    team_pos.columns = team_pos.columns.droplevel(0)
    team_pos = team_pos[['Touches', 'Def 3rd', 'Mid 3rd', 'Att 3rd', 'Succ%', '1/3']]
    team_pos.columns = ['Touches', 'Touches_Def_3rd', 'Touches_Mid_3rd', 'Touches_Att_3rd', 'Take_On_Success_%', 'Carries_into_Final_3rd']
    team_pos = team_pos.rename(index={len(team_pos)-1: min(teams, key=teams.get)})
    team_pos = team_pos.iloc[-1:]
    #team_pos

    team_misc = pd.read_html(link)[8]
    team_misc.columns = team_misc.columns.droplevel(0)
    team_misc = team_misc[['Recov', 'Won%']]
    team_misc.columns = ['Ball_Recoveries', 'Aerial_Duels_%']
    team_misc = team_misc.rename(index={len(team_misc)-1: min(teams, key=teams.get)})
    team_misc = team_misc.iloc[-1:]
    #team_misc

    team_gk = pd.read_html(link)[9]
    team_gk.columns = team_gk.columns.droplevel(0)
    team_gk = team_gk[['PSxG', '#OPA', 'AvgDist']]
    team_gk.columns = ['Post_Shot_xG', 'Def_Actions_Outside_Area', 'Avg_Dist_Actions_Outside_Area']
    team_gk = team_gk.rename(index={len(team_gk)-1: min(teams, key=teams.get)})
    team_gk = team_gk.iloc[-1:]
    #team_gk

    team_dfs = [team_stand, team_pass, team_def, team_pos, team_misc, team_gk]
    team_merge = reduce(lambda  left, right: pd.merge(left, right, left_index=True, right_index=True), team_dfs)
    team_merge = team_merge.apply(pd.to_numeric)
    #team_merge

    team2_stand = pd.read_html(link)[10]
    team2_stand.columns = team2_stand.columns.droplevel(0)
    team2_stand = team2_stand[['Gls', 'Ast', 'Sh', 'SoT', 'xG', 'npxG', 'SCA', 'GCA', 'PrgP', 'PrgC']]
    team2_stand.columns = ['Goals', 'Assists', 'Shots', 'Shots_on_Target', 'xG', 'NPxG', 'SCA', 'GCA', 'Progressive_Passes', 'Progressive_Carries']
    team2_stand = team2_stand.rename(index={len(team2_stand)-1: max(teams, key=teams.get)})
    team2_stand = team2_stand.iloc[-1:]
    #team2_stand

    team2_pass = pd.read_html(link)[11]
    team2_pass.columns = team2_pass.columns.droplevel(0)
    passing = ["Player", "#", "Pos", "Age", "Min", "passes_completed", "passes", "passes_pct", "passes_total_distance","passes_progressive_distance","passes_completed_short","passes_short","passes_pct_short","passes_completed_medium","passes_medium","passes_pct_medium","passes_completed_long","passes_long","passes_pct_long","assists","xa_chain","xa","key_passes","passes_into_final_third","passes_into_penalty_area","crosses_into_penalty_area","progressive_passes"]
    team2_pass.columns = passing
    team2_pass = team2_pass.iloc[:, [9, 10, 12, 13, 15, 16, 18, 21, 25]]
    team2_pass.columns = ['Progressive_Pass_Dist', 'Short_Passes_Completed', 'Short_Pass_%','Medium_Passes_Completed','Medium_Pass_%','Long_Passes_Completed','Long_Pass_%','xA', 'Crosses_into_Penalty_Area']
    team2_pass = team2_pass.rename(index={len(team2_pass)-1: max(teams, key=teams.get)})
    team2_pass = team2_pass.iloc[-1:]
    #team2_pass

    team2_def = pd.read_html(link)[13]
    team2_def.columns = team2_def.columns.droplevel(0)
    team2_def = team2_def[['Tkl', 'TklW', 'Blocks', 'Int', 'Clr']]
    team2_def.columns = ['Tackles', 'Challenges', 'Tackles_Won', 'Blocks', 'Interceptions', 'Clearances']
    team2_def = team2_def.drop(['Challenges'], axis=1)
    team2_def = team2_def.rename(index={len(team2_def)-1: max(teams, key=teams.get)})
    team2_def = team2_def.iloc[-1:]
    #team2_def

    team2_pos = pd.read_html(link)[14]
    team2_pos.columns = team2_pos.columns.droplevel(0)
    team2_pos = team2_pos[['Touches', 'Def 3rd', 'Mid 3rd', 'Att 3rd', 'Succ%', '1/3']]
    team2_pos.columns = ['Touches', 'Touches_Def_3rd', 'Touches_Mid_3rd', 'Touches_Att_3rd', 'Take_On_Success_%', 'Carries_into_Final_3rd']
    team2_pos = team2_pos.rename(index={len(team2_pos)-1: max(teams, key=teams.get)})
    team2_pos = team2_pos.iloc[-1:]
    #team2_pos

    team2_misc = pd.read_html(link)[15]
    team2_misc.columns = team2_misc.columns.droplevel(0)
    team2_misc = team2_misc[['Recov', 'Won%']]
    team2_misc.columns = ['Ball_Recoveries', 'Aerial_Duels_%']
    team2_misc = team2_misc.rename(index={len(team2_misc)-1: max(teams, key=teams.get)})
    team2_misc = team2_misc.iloc[-1:]
    #team2_misc

    team2_gk = pd.read_html(link)[16]
    team2_gk.columns = team2_gk.columns.droplevel(0)
    team2_gk = team2_gk[['PSxG', '#OPA', 'AvgDist']]
    team2_gk.columns = ['Post_Shot_xG', 'Def_Actions_Outside_Area', 'Avg_Dist_Actions_Outside_Area']
    team2_gk = team2_gk.rename(index={len(team2_gk)-1: max(teams, key=teams.get)})
    team2_gk = team2_gk.iloc[-1:]
    #team2_gk

    team2_dfs = [team2_stand, team2_pass, team2_def, team2_pos, team2_misc, team2_gk]
    team2_merge = reduce(lambda  left, right: pd.merge(left, right, left_index=True, right_index=True), team2_dfs)
    team2_merge = team2_merge.apply(pd.to_numeric)
    #team2_merge

    if (team_merge['Goals'][0] > team2_merge['Goals'][0]) == True:
        team_merge['Result'] = 3
        team2_merge['Result'] = 0
    elif (team_merge['Goals'][0] == team2_merge['Goals'][0]) == True:
        team_merge['Result'] = 1
        team2_merge['Result'] = 1
    elif (team_merge['Goals'][0] < team2_merge['Goals'][0]) == True:
        team_merge['Result'] = 0
        team2_merge['Result'] = 3;
    #team_merge
    #team2_merge

    df1_name = (min(teams, key=teams.get) + link.split(max(teams, key=teams.get), 1)[1]).replace("-", "_")
    team_merge = team_merge.rename(index={min(teams, key=teams.get): df1_name})
    #team_merge

    df2_name = (max(teams, key=teams.get) + link.split(max(teams, key=teams.get), 1)[1]).replace("-", "_")
    team2_merge = team2_merge.rename(index={max(teams, key=teams.get): df2_name})
    #team2_merge

    #ca19_game1 = pd.concat([team_merge, team2_merge], axis=0)
    time.sleep(60)
    global ca21
    ca21 = pd.concat([ca21, team_merge, team2_merge], axis=0)
    #return (team_merge, team2_merge)


for i in range(0, len(ca21_games)):
    ca21_match_data(ca21_games[i]);

ca21.to_csv("data_tables\copa21.csv", index=True)


###Obtaining world cup 2018 match data
wc18_countries = ['Argentina', 'Australia', 'Belgium', 'Brazil', 'Colombia', 'Costa-Rica', 'Croatia', 'Denmark', 'Egypt', 'England', 'France', 'Germany', 'Iceland', 'IR-Iran', 'Japan', 'Korea-Republic', 'Mexico', 'Morocco', 'Nigeria', 'Panama', 'Peru', 'Poland', 'Portugal', 'Russia', 'Saudi-Arabia', 'Senegal', 'Serbia', 'Spain', 'Sweden', 'Switzerland', 'Tunisia', 'Uruguay']

wc18_url = 'https://fbref.com/en/comps/1/2018/schedule/2018-World-Cup-Scores-and-Fixtures'
reqs = requests.get(wc18_url)
soup = BeautifulSoup(reqs.text, 'html.parser')

wc18_urls = []
for link in soup.find_all('a'):
    wc18_urls.append(link.get('href'))

wc18_urls_str = [str(i) for i in wc18_urls]
wc18_games = [s for s in wc18_urls_str if any(k in s for k in wc18_countries)]
wc18_games = [s for s in wc18_games if "2018-World-Cup" in s]
wc18_games = list(dict.fromkeys(wc18_games))
print(wc18_games)
len(wc18_games) == 64

wc18_games = ['https://fbref.com' + i for i in wc18_games]

wc18 = pd.DataFrame()
def wc18_match_data(link):

    teams = {}
    for i in range(0, len(wc18_countries)):
        if link.find(wc18_countries[i]) > 0:
            teams.update({wc18_countries[i]: link.find(wc18_countries[i])});
    #teams
    #print(min(teams, key=teams.get))

    team_stand = pd.read_html(link)[3]
    team_stand.columns = team_stand.columns.droplevel(0)
    team_stand = team_stand[['Gls', 'Ast', 'Sh', 'SoT', 'xG', 'npxG', 'SCA', 'GCA', 'PrgP', 'PrgC']]
    team_stand.columns = ['Goals', 'Assists', 'Shots', 'Shots_on_Target', 'xG', 'NPxG', 'SCA', 'GCA', 'Progressive_Passes', 'Progressive_Carries']
    team_stand = team_stand.rename(index={len(team_stand)-1: min(teams, key=teams.get)})
    team_stand = team_stand.iloc[-1:]
    #team_stand

    team_pass = pd.read_html(link)[4]
    team_pass.columns = team_pass.columns.droplevel(0)
    passing = ["Player", "#", "Pos", "Age", "Min", "passes_completed", "passes", "passes_pct", "passes_total_distance","passes_progressive_distance","passes_completed_short","passes_short","passes_pct_short","passes_completed_medium","passes_medium","passes_pct_medium","passes_completed_long","passes_long","passes_pct_long","assists","xa_chain","xa","key_passes","passes_into_final_third","passes_into_penalty_area","crosses_into_penalty_area","progressive_passes"]
    team_pass.columns = passing
    team_pass = team_pass.iloc[:, [9, 10, 12, 13, 15, 16, 18, 21, 25]]
    team_pass.columns = ['Progressive_Pass_Dist', 'Short_Passes_Completed', 'Short_Pass_%','Medium_Passes_Completed','Medium_Pass_%','Long_Passes_Completed','Long_Pass_%','xA', 'Crosses_into_Penalty_Area']
    team_pass = team_pass.rename(index={len(team_pass)-1: min(teams, key=teams.get)})
    team_pass = team_pass.iloc[-1:]
    #team_pass

    team_def = pd.read_html(link)[6]
    team_def.columns = team_def.columns.droplevel(0)
    team_def = team_def[['Tkl', 'TklW', 'Blocks', 'Int', 'Clr']]
    team_def.columns = ['Tackles', 'Challenges', 'Tackles_Won', 'Blocks', 'Interceptions', 'Clearances']
    team_def = team_def.drop(['Challenges'], axis=1)
    team_def = team_def.rename(index={len(team_def)-1: min(teams, key=teams.get)})
    team_def = team_def.iloc[-1:]
    #team_def

    team_pos = pd.read_html(link)[7]
    team_pos.columns = team_pos.columns.droplevel(0)
    team_pos = team_pos[['Touches', 'Def 3rd', 'Mid 3rd', 'Att 3rd', 'Succ%', '1/3']]
    team_pos.columns = ['Touches', 'Touches_Def_3rd', 'Touches_Mid_3rd', 'Touches_Att_3rd', 'Take_On_Success_%', 'Carries_into_Final_3rd']
    team_pos = team_pos.rename(index={len(team_pos)-1: min(teams, key=teams.get)})
    team_pos = team_pos.iloc[-1:]
    #team_pos

    team_misc = pd.read_html(link)[8]
    team_misc.columns = team_misc.columns.droplevel(0)
    team_misc = team_misc[['Recov', 'Won%']]
    team_misc.columns = ['Ball_Recoveries', 'Aerial_Duels_%']
    team_misc = team_misc.rename(index={len(team_misc)-1: min(teams, key=teams.get)})
    team_misc = team_misc.iloc[-1:]
    #team_misc

    team_gk = pd.read_html(link)[9]
    team_gk.columns = team_gk.columns.droplevel(0)
    team_gk = team_gk[['PSxG', '#OPA', 'AvgDist']]
    team_gk.columns = ['Post_Shot_xG', 'Def_Actions_Outside_Area', 'Avg_Dist_Actions_Outside_Area']
    team_gk = team_gk.rename(index={len(team_gk)-1: min(teams, key=teams.get)})
    team_gk = team_gk.iloc[-1:]
    #team_gk

    team_dfs = [team_stand, team_pass, team_def, team_pos, team_misc, team_gk]
    team_merge = reduce(lambda  left, right: pd.merge(left, right, left_index=True, right_index=True), team_dfs)
    team_merge = team_merge.apply(pd.to_numeric)
    #team_merge

    team2_stand = pd.read_html(link)[10]
    team2_stand.columns = team2_stand.columns.droplevel(0)
    team2_stand = team2_stand[['Gls', 'Ast', 'Sh', 'SoT', 'xG', 'npxG', 'SCA', 'GCA', 'PrgP', 'PrgC']]
    team2_stand.columns = ['Goals', 'Assists', 'Shots', 'Shots_on_Target', 'xG', 'NPxG', 'SCA', 'GCA', 'Progressive_Passes', 'Progressive_Carries']
    team2_stand = team2_stand.rename(index={len(team2_stand)-1: max(teams, key=teams.get)})
    team2_stand = team2_stand.iloc[-1:]
    #team2_stand

    team2_pass = pd.read_html(link)[11]
    team2_pass.columns = team2_pass.columns.droplevel(0)
    passing = ["Player", "#", "Pos", "Age", "Min", "passes_completed", "passes", "passes_pct", "passes_total_distance","passes_progressive_distance","passes_completed_short","passes_short","passes_pct_short","passes_completed_medium","passes_medium","passes_pct_medium","passes_completed_long","passes_long","passes_pct_long","assists","xa_chain","xa","key_passes","passes_into_final_third","passes_into_penalty_area","crosses_into_penalty_area","progressive_passes"]
    team2_pass.columns = passing
    team2_pass = team2_pass.iloc[:, [9, 10, 12, 13, 15, 16, 18, 21, 25]]
    team2_pass.columns = ['Progressive_Pass_Dist', 'Short_Passes_Completed', 'Short_Pass_%','Medium_Passes_Completed','Medium_Pass_%','Long_Passes_Completed','Long_Pass_%','xA', 'Crosses_into_Penalty_Area']
    team2_pass = team2_pass.rename(index={len(team2_pass)-1: max(teams, key=teams.get)})
    team2_pass = team2_pass.iloc[-1:]
    #team2_pass

    team2_def = pd.read_html(link)[13]
    team2_def.columns = team2_def.columns.droplevel(0)
    team2_def = team2_def[['Tkl', 'TklW', 'Blocks', 'Int', 'Clr']]
    team2_def.columns = ['Tackles', 'Challenges', 'Tackles_Won', 'Blocks', 'Interceptions', 'Clearances']
    team2_def = team2_def.drop(['Challenges'], axis=1)
    team2_def = team2_def.rename(index={len(team2_def)-1: max(teams, key=teams.get)})
    team2_def = team2_def.iloc[-1:]
    #team2_def

    team2_pos = pd.read_html(link)[14]
    team2_pos.columns = team2_pos.columns.droplevel(0)
    team2_pos = team2_pos[['Touches', 'Def 3rd', 'Mid 3rd', 'Att 3rd', 'Succ%', '1/3']]
    team2_pos.columns = ['Touches', 'Touches_Def_3rd', 'Touches_Mid_3rd', 'Touches_Att_3rd', 'Take_On_Success_%', 'Carries_into_Final_3rd']
    team2_pos = team2_pos.rename(index={len(team2_pos)-1: max(teams, key=teams.get)})
    team2_pos = team2_pos.iloc[-1:]
    #team2_pos

    team2_misc = pd.read_html(link)[15]
    team2_misc.columns = team2_misc.columns.droplevel(0)
    team2_misc = team2_misc[['Recov', 'Won%']]
    team2_misc.columns = ['Ball_Recoveries', 'Aerial_Duels_%']
    team2_misc = team2_misc.rename(index={len(team2_misc)-1: max(teams, key=teams.get)})
    team2_misc = team2_misc.iloc[-1:]
    #team2_misc

    team2_gk = pd.read_html(link)[16]
    team2_gk.columns = team2_gk.columns.droplevel(0)
    team2_gk = team2_gk[['PSxG', '#OPA', 'AvgDist']]
    team2_gk.columns = ['Post_Shot_xG', 'Def_Actions_Outside_Area', 'Avg_Dist_Actions_Outside_Area']
    team2_gk = team2_gk.rename(index={len(team2_gk)-1: max(teams, key=teams.get)})
    team2_gk = team2_gk.iloc[-1:]
    #team2_gk

    team2_dfs = [team2_stand, team2_pass, team2_def, team2_pos, team2_misc, team2_gk]
    team2_merge = reduce(lambda  left, right: pd.merge(left, right, left_index=True, right_index=True), team2_dfs)
    team2_merge = team2_merge.apply(pd.to_numeric)
    #team2_merge

    if (team_merge['Goals'][0] > team2_merge['Goals'][0]) == True:
        team_merge['Result'] = 3
        team2_merge['Result'] = 0
    elif (team_merge['Goals'][0] == team2_merge['Goals'][0]) == True:
        team_merge['Result'] = 1
        team2_merge['Result'] = 1
    elif (team_merge['Goals'][0] < team2_merge['Goals'][0]) == True:
        team_merge['Result'] = 0
        team2_merge['Result'] = 3;
    #team_merge
    #team2_merge

    df1_name = (min(teams, key=teams.get) + link.split(max(teams, key=teams.get), 1)[1]).replace("-", "_")
    team_merge = team_merge.rename(index={min(teams, key=teams.get): df1_name})
    #team_merge

    df2_name = (max(teams, key=teams.get) + link.split(max(teams, key=teams.get), 1)[1]).replace("-", "_")
    team2_merge = team2_merge.rename(index={max(teams, key=teams.get): df2_name})
    #team2_merge

    #ca19_game1 = pd.concat([team_merge, team2_merge], axis=0)
    time.sleep(60)
    global wc18
    wc18 = pd.concat([wc18, team_merge, team2_merge], axis=0)
    #return (team_merge, team2_merge)


for i in range(0, len(wc18_games)):
    wc18_match_data(wc18_games[i]);

wc18.to_csv("data_tables\worldc18.csv", index=True)


###Obtaining world cup 2022 match data
wc22_countries = ['Argentina', 'Australia', 'Belgium', 'Brazil', 'Cameroon', 'Canada', 'Costa-Rica', 'Croatia', 'Denmark', 'Ecuador', 'England', 'France', 'Germany', 'Ghana', 'IR-Iran', 'Japan', 'Korea-Republic', 'Mexico', 'Morocco', 'Netherlands', 'Poland', 'Portugal', 'Qatar', 'Saudi-Arabia', 'Senegal', 'Serbia', 'Spain', 'Switzerland', 'Tunisia', 'United-States', 'Uruguay', 'Wales']

wc22_url = 'https://fbref.com/en/comps/1/2022/schedule/2022-World-Cup-Scores-and-Fixtures'
reqs = requests.get(wc22_url)
soup = BeautifulSoup(reqs.text, 'html.parser')

wc22_urls = []
for link in soup.find_all('a'):
    wc22_urls.append(link.get('href'))

wc22_urls_str = [str(i) for i in wc22_urls]
wc22_games = [s for s in wc22_urls_str if any(k in s for k in wc22_countries)]
wc22_games = [s for s in wc22_games if "2022-World-Cup" in s]
wc22_games = list(dict.fromkeys(wc22_games))
print(wc22_games)
len(wc22_games) == 64

wc22_games = ['https://fbref.com' + i for i in wc22_games]

wc22 = pd.DataFrame()
def wc22_match_data(link):

    teams = {}
    for i in range(0, len(wc22_countries)):
        if link.find(wc22_countries[i]) > 0:
            teams.update({wc22_countries[i]: link.find(wc22_countries[i])});
    #teams
    #print(min(teams, key=teams.get))

    team_stand = pd.read_html(link)[3]
    team_stand.columns = team_stand.columns.droplevel(0)
    team_stand = team_stand[['Gls', 'Ast', 'Sh', 'SoT', 'xG', 'npxG', 'SCA', 'GCA', 'PrgP', 'PrgC']]
    team_stand.columns = ['Goals', 'Assists', 'Shots', 'Shots_on_Target', 'xG', 'NPxG', 'SCA', 'GCA', 'Progressive_Passes', 'Progressive_Carries']
    team_stand = team_stand.rename(index={len(team_stand)-1: min(teams, key=teams.get)})
    team_stand = team_stand.iloc[-1:]
    #team_stand

    team_pass = pd.read_html(link)[4]
    team_pass.columns = team_pass.columns.droplevel(0)
    passing = ["Player", "#", "Pos", "Age", "Min", "passes_completed", "passes", "passes_pct", "passes_total_distance","passes_progressive_distance","passes_completed_short","passes_short","passes_pct_short","passes_completed_medium","passes_medium","passes_pct_medium","passes_completed_long","passes_long","passes_pct_long","assists","xa_chain","xa","key_passes","passes_into_final_third","passes_into_penalty_area","crosses_into_penalty_area","progressive_passes"]
    team_pass.columns = passing
    team_pass = team_pass.iloc[:, [9, 10, 12, 13, 15, 16, 18, 21, 25]]
    team_pass.columns = ['Progressive_Pass_Dist', 'Short_Passes_Completed', 'Short_Pass_%','Medium_Passes_Completed','Medium_Pass_%','Long_Passes_Completed','Long_Pass_%','xA', 'Crosses_into_Penalty_Area']
    team_pass = team_pass.rename(index={len(team_pass)-1: min(teams, key=teams.get)})
    team_pass = team_pass.iloc[-1:]
    #team_pass

    team_def = pd.read_html(link)[6]
    team_def.columns = team_def.columns.droplevel(0)
    team_def = team_def[['Tkl', 'TklW', 'Blocks', 'Int', 'Clr']]
    team_def.columns = ['Tackles', 'Challenges', 'Tackles_Won', 'Blocks', 'Interceptions', 'Clearances']
    team_def = team_def.drop(['Challenges'], axis=1)
    team_def = team_def.rename(index={len(team_def)-1: min(teams, key=teams.get)})
    team_def = team_def.iloc[-1:]
    #team_def

    team_pos = pd.read_html(link)[7]
    team_pos.columns = team_pos.columns.droplevel(0)
    team_pos = team_pos[['Touches', 'Def 3rd', 'Mid 3rd', 'Att 3rd', 'Succ%', '1/3']]
    team_pos.columns = ['Touches', 'Touches_Def_3rd', 'Touches_Mid_3rd', 'Touches_Att_3rd', 'Take_On_Success_%', 'Carries_into_Final_3rd']
    team_pos = team_pos.rename(index={len(team_pos)-1: min(teams, key=teams.get)})
    team_pos = team_pos.iloc[-1:]
    #team_pos

    team_misc = pd.read_html(link)[8]
    team_misc.columns = team_misc.columns.droplevel(0)
    team_misc = team_misc[['Recov', 'Won%']]
    team_misc.columns = ['Ball_Recoveries', 'Aerial_Duels_%']
    team_misc = team_misc.rename(index={len(team_misc)-1: min(teams, key=teams.get)})
    team_misc = team_misc.iloc[-1:]
    #team_misc

    team_gk = pd.read_html(link)[9]
    team_gk.columns = team_gk.columns.droplevel(0)
    team_gk = team_gk[['PSxG', '#OPA', 'AvgDist']]
    team_gk.columns = ['Post_Shot_xG', 'Def_Actions_Outside_Area', 'Avg_Dist_Actions_Outside_Area']
    team_gk = team_gk.rename(index={len(team_gk)-1: min(teams, key=teams.get)})
    team_gk = team_gk.iloc[-1:]
    #team_gk

    team_dfs = [team_stand, team_pass, team_def, team_pos, team_misc, team_gk]
    team_merge = reduce(lambda  left, right: pd.merge(left, right, left_index=True, right_index=True), team_dfs)
    team_merge = team_merge.apply(pd.to_numeric)
    #team_merge

    team2_stand = pd.read_html(link)[10]
    team2_stand.columns = team2_stand.columns.droplevel(0)
    team2_stand = team2_stand[['Gls', 'Ast', 'Sh', 'SoT', 'xG', 'npxG', 'SCA', 'GCA', 'PrgP', 'PrgC']]
    team2_stand.columns = ['Goals', 'Assists', 'Shots', 'Shots_on_Target', 'xG', 'NPxG', 'SCA', 'GCA', 'Progressive_Passes', 'Progressive_Carries']
    team2_stand = team2_stand.rename(index={len(team2_stand)-1: max(teams, key=teams.get)})
    team2_stand = team2_stand.iloc[-1:]
    #team2_stand

    team2_pass = pd.read_html(link)[11]
    team2_pass.columns = team2_pass.columns.droplevel(0)
    passing = ["Player", "#", "Pos", "Age", "Min", "passes_completed", "passes", "passes_pct", "passes_total_distance","passes_progressive_distance","passes_completed_short","passes_short","passes_pct_short","passes_completed_medium","passes_medium","passes_pct_medium","passes_completed_long","passes_long","passes_pct_long","assists","xa_chain","xa","key_passes","passes_into_final_third","passes_into_penalty_area","crosses_into_penalty_area","progressive_passes"]
    team2_pass.columns = passing
    team2_pass = team2_pass.iloc[:, [9, 10, 12, 13, 15, 16, 18, 21, 25]]
    team2_pass.columns = ['Progressive_Pass_Dist', 'Short_Passes_Completed', 'Short_Pass_%','Medium_Passes_Completed','Medium_Pass_%','Long_Passes_Completed','Long_Pass_%','xA', 'Crosses_into_Penalty_Area']
    team2_pass = team2_pass.rename(index={len(team2_pass)-1: max(teams, key=teams.get)})
    team2_pass = team2_pass.iloc[-1:]
    #team2_pass

    team2_def = pd.read_html(link)[13]
    team2_def.columns = team2_def.columns.droplevel(0)
    team2_def = team2_def[['Tkl', 'TklW', 'Blocks', 'Int', 'Clr']]
    team2_def.columns = ['Tackles', 'Challenges', 'Tackles_Won', 'Blocks', 'Interceptions', 'Clearances']
    team2_def = team2_def.drop(['Challenges'], axis=1)
    team2_def = team2_def.rename(index={len(team2_def)-1: max(teams, key=teams.get)})
    team2_def = team2_def.iloc[-1:]
    #team2_def

    team2_pos = pd.read_html(link)[14]
    team2_pos.columns = team2_pos.columns.droplevel(0)
    team2_pos = team2_pos[['Touches', 'Def 3rd', 'Mid 3rd', 'Att 3rd', 'Succ%', '1/3']]
    team2_pos.columns = ['Touches', 'Touches_Def_3rd', 'Touches_Mid_3rd', 'Touches_Att_3rd', 'Take_On_Success_%', 'Carries_into_Final_3rd']
    team2_pos = team2_pos.rename(index={len(team2_pos)-1: max(teams, key=teams.get)})
    team2_pos = team2_pos.iloc[-1:]
    #team2_pos

    team2_misc = pd.read_html(link)[15]
    team2_misc.columns = team2_misc.columns.droplevel(0)
    team2_misc = team2_misc[['Recov', 'Won%']]
    team2_misc.columns = ['Ball_Recoveries', 'Aerial_Duels_%']
    team2_misc = team2_misc.rename(index={len(team2_misc)-1: max(teams, key=teams.get)})
    team2_misc = team2_misc.iloc[-1:]
    #team2_misc

    team2_gk = pd.read_html(link)[16]
    team2_gk.columns = team2_gk.columns.droplevel(0)
    team2_gk = team2_gk[['PSxG', '#OPA', 'AvgDist']]
    team2_gk.columns = ['Post_Shot_xG', 'Def_Actions_Outside_Area', 'Avg_Dist_Actions_Outside_Area']
    team2_gk = team2_gk.rename(index={len(team2_gk)-1: max(teams, key=teams.get)})
    team2_gk = team2_gk.iloc[-1:]
    #team2_gk

    team2_dfs = [team2_stand, team2_pass, team2_def, team2_pos, team2_misc, team2_gk]
    team2_merge = reduce(lambda  left, right: pd.merge(left, right, left_index=True, right_index=True), team2_dfs)
    team2_merge = team2_merge.apply(pd.to_numeric)
    #team2_merge

    if (team_merge['Goals'][0] > team2_merge['Goals'][0]) == True:
        team_merge['Result'] = 3
        team2_merge['Result'] = 0
    elif (team_merge['Goals'][0] == team2_merge['Goals'][0]) == True:
        team_merge['Result'] = 1
        team2_merge['Result'] = 1
    elif (team_merge['Goals'][0] < team2_merge['Goals'][0]) == True:
        team_merge['Result'] = 0
        team2_merge['Result'] = 3;
    #team_merge
    #team2_merge

    df1_name = (min(teams, key=teams.get) + link.split(max(teams, key=teams.get), 1)[1]).replace("-", "_")
    team_merge = team_merge.rename(index={min(teams, key=teams.get): df1_name})
    #team_merge

    df2_name = (max(teams, key=teams.get) + link.split(max(teams, key=teams.get), 1)[1]).replace("-", "_")
    team2_merge = team2_merge.rename(index={max(teams, key=teams.get): df2_name})
    #team2_merge

    #ca19_game1 = pd.concat([team_merge, team2_merge], axis=0)
    time.sleep(60)
    global wc22
    wc22 = pd.concat([wc22, team_merge, team2_merge], axis=0)
    #return (team_merge, team2_merge)


for i in range(0, len(wc22_games)):
    wc22_match_data(wc22_games[i]);

wc22.to_csv("data_tables\worldc22.csv", index=True)


###Obtaining euro 2021 match data
e21_countries = ['Austria', 'Belgium', 'Croatia', 'Czechia', 'Denmark', 'England', 'Finland', 'France', 'Germany', 'Hungary', 'Italy', 'North-Macedonia', 'Netherlands', 'Poland', 'Portugal', 'Russia', 'Scotland', 'Slovakia', 'Spain', 'Sweden', 'Switzerland', 'Turkiye', 'Ukraine', 'Wales']

e21_url = 'https://fbref.com/en/comps/676/2021/schedule/2021-European-Championship-Scores-and-Fixtures'
reqs = requests.get(e21_url)
soup = BeautifulSoup(reqs.text, 'html.parser')

e21_urls = []
for link in soup.find_all('a'):
    e21_urls.append(link.get('href'))

e21_urls_str = [str(i) for i in e21_urls]
e21_games = [s for s in e21_urls_str if any(k in s for k in e21_countries)]
e21_games = [s for s in e21_games if "2021-European-Championship" in s]
e21_games = list(dict.fromkeys(e21_games))
print(e21_games)
len(e21_games) == 51

e21_games = ['https://fbref.com' + i for i in e21_games]

e21 = pd.DataFrame()
def e21_match_data(link):

    teams = {}
    for i in range(0, len(e21_countries)):
        if link.find(e21_countries[i]) > 0:
            teams.update({e21_countries[i]: link.find(e21_countries[i])});
    #teams
    #print(min(teams, key=teams.get))

    team_stand = pd.read_html(link)[3]
    team_stand.columns = team_stand.columns.droplevel(0)
    team_stand = team_stand[['Gls', 'Ast', 'Sh', 'SoT', 'xG', 'npxG', 'SCA', 'GCA', 'PrgP', 'PrgC']]
    team_stand.columns = ['Goals', 'Assists', 'Shots', 'Shots_on_Target', 'xG', 'NPxG', 'SCA', 'GCA', 'Progressive_Passes', 'Progressive_Carries']
    team_stand = team_stand.rename(index={len(team_stand)-1: min(teams, key=teams.get)})
    team_stand = team_stand.iloc[-1:]
    #team_stand

    team_pass = pd.read_html(link)[4]
    team_pass.columns = team_pass.columns.droplevel(0)
    passing = ["Player", "#", "Pos", "Age", "Min", "passes_completed", "passes", "passes_pct", "passes_total_distance","passes_progressive_distance","passes_completed_short","passes_short","passes_pct_short","passes_completed_medium","passes_medium","passes_pct_medium","passes_completed_long","passes_long","passes_pct_long","assists","xa_chain","xa","key_passes","passes_into_final_third","passes_into_penalty_area","crosses_into_penalty_area","progressive_passes"]
    team_pass.columns = passing
    team_pass = team_pass.iloc[:, [9, 10, 12, 13, 15, 16, 18, 21, 25]]
    team_pass.columns = ['Progressive_Pass_Dist', 'Short_Passes_Completed', 'Short_Pass_%','Medium_Passes_Completed','Medium_Pass_%','Long_Passes_Completed','Long_Pass_%','xA', 'Crosses_into_Penalty_Area']
    team_pass = team_pass.rename(index={len(team_pass)-1: min(teams, key=teams.get)})
    team_pass = team_pass.iloc[-1:]
    #team_pass

    team_def = pd.read_html(link)[6]
    team_def.columns = team_def.columns.droplevel(0)
    team_def = team_def[['Tkl', 'TklW', 'Blocks', 'Int', 'Clr']]
    team_def.columns = ['Tackles', 'Challenges', 'Tackles_Won', 'Blocks', 'Interceptions', 'Clearances']
    team_def = team_def.drop(['Challenges'], axis=1)
    team_def = team_def.rename(index={len(team_def)-1: min(teams, key=teams.get)})
    team_def = team_def.iloc[-1:]
    #team_def

    team_pos = pd.read_html(link)[7]
    team_pos.columns = team_pos.columns.droplevel(0)
    team_pos = team_pos[['Touches', 'Def 3rd', 'Mid 3rd', 'Att 3rd', 'Succ%', '1/3']]
    team_pos.columns = ['Touches', 'Touches_Def_3rd', 'Touches_Mid_3rd', 'Touches_Att_3rd', 'Take_On_Success_%', 'Carries_into_Final_3rd']
    team_pos = team_pos.rename(index={len(team_pos)-1: min(teams, key=teams.get)})
    team_pos = team_pos.iloc[-1:]
    #team_pos

    team_misc = pd.read_html(link)[8]
    team_misc.columns = team_misc.columns.droplevel(0)
    team_misc = team_misc[['Recov', 'Won%']]
    team_misc.columns = ['Ball_Recoveries', 'Aerial_Duels_%']
    team_misc = team_misc.rename(index={len(team_misc)-1: min(teams, key=teams.get)})
    team_misc = team_misc.iloc[-1:]
    #team_misc

    team_gk = pd.read_html(link)[9]
    team_gk.columns = team_gk.columns.droplevel(0)
    team_gk = team_gk[['PSxG', '#OPA', 'AvgDist']]
    team_gk.columns = ['Post_Shot_xG', 'Def_Actions_Outside_Area', 'Avg_Dist_Actions_Outside_Area']
    team_gk = team_gk.rename(index={len(team_gk)-1: min(teams, key=teams.get)})
    team_gk = team_gk.iloc[-1:]
    #team_gk

    team_dfs = [team_stand, team_pass, team_def, team_pos, team_misc, team_gk]
    team_merge = reduce(lambda  left, right: pd.merge(left, right, left_index=True, right_index=True), team_dfs)
    team_merge = team_merge.apply(pd.to_numeric)
    #team_merge

    team2_stand = pd.read_html(link)[10]
    team2_stand.columns = team2_stand.columns.droplevel(0)
    team2_stand = team2_stand[['Gls', 'Ast', 'Sh', 'SoT', 'xG', 'npxG', 'SCA', 'GCA', 'PrgP', 'PrgC']]
    team2_stand.columns = ['Goals', 'Assists', 'Shots', 'Shots_on_Target', 'xG', 'NPxG', 'SCA', 'GCA', 'Progressive_Passes', 'Progressive_Carries']
    team2_stand = team2_stand.rename(index={len(team2_stand)-1: max(teams, key=teams.get)})
    team2_stand = team2_stand.iloc[-1:]
    #team2_stand

    team2_pass = pd.read_html(link)[11]
    team2_pass.columns = team2_pass.columns.droplevel(0)
    passing = ["Player", "#", "Pos", "Age", "Min", "passes_completed", "passes", "passes_pct", "passes_total_distance","passes_progressive_distance","passes_completed_short","passes_short","passes_pct_short","passes_completed_medium","passes_medium","passes_pct_medium","passes_completed_long","passes_long","passes_pct_long","assists","xa_chain","xa","key_passes","passes_into_final_third","passes_into_penalty_area","crosses_into_penalty_area","progressive_passes"]
    team2_pass.columns = passing
    team2_pass = team2_pass.iloc[:, [9, 10, 12, 13, 15, 16, 18, 21, 25]]
    team2_pass.columns = ['Progressive_Pass_Dist', 'Short_Passes_Completed', 'Short_Pass_%','Medium_Passes_Completed','Medium_Pass_%','Long_Passes_Completed','Long_Pass_%','xA', 'Crosses_into_Penalty_Area']
    team2_pass = team2_pass.rename(index={len(team2_pass)-1: max(teams, key=teams.get)})
    team2_pass = team2_pass.iloc[-1:]
    #team2_pass

    team2_def = pd.read_html(link)[13]
    team2_def.columns = team2_def.columns.droplevel(0)
    team2_def = team2_def[['Tkl', 'TklW', 'Blocks', 'Int', 'Clr']]
    team2_def.columns = ['Tackles', 'Challenges', 'Tackles_Won', 'Blocks', 'Interceptions', 'Clearances']
    team2_def = team2_def.drop(['Challenges'], axis=1)
    team2_def = team2_def.rename(index={len(team2_def)-1: max(teams, key=teams.get)})
    team2_def = team2_def.iloc[-1:]
    #team2_def

    team2_pos = pd.read_html(link)[14]
    team2_pos.columns = team2_pos.columns.droplevel(0)
    team2_pos = team2_pos[['Touches', 'Def 3rd', 'Mid 3rd', 'Att 3rd', 'Succ%', '1/3']]
    team2_pos.columns = ['Touches', 'Touches_Def_3rd', 'Touches_Mid_3rd', 'Touches_Att_3rd', 'Take_On_Success_%', 'Carries_into_Final_3rd']
    team2_pos = team2_pos.rename(index={len(team2_pos)-1: max(teams, key=teams.get)})
    team2_pos = team2_pos.iloc[-1:]
    #team2_pos

    team2_misc = pd.read_html(link)[15]
    team2_misc.columns = team2_misc.columns.droplevel(0)
    team2_misc = team2_misc[['Recov', 'Won%']]
    team2_misc.columns = ['Ball_Recoveries', 'Aerial_Duels_%']
    team2_misc = team2_misc.rename(index={len(team2_misc)-1: max(teams, key=teams.get)})
    team2_misc = team2_misc.iloc[-1:]
    #team2_misc

    team2_gk = pd.read_html(link)[16]
    team2_gk.columns = team2_gk.columns.droplevel(0)
    team2_gk = team2_gk[['PSxG', '#OPA', 'AvgDist']]
    team2_gk.columns = ['Post_Shot_xG', 'Def_Actions_Outside_Area', 'Avg_Dist_Actions_Outside_Area']
    team2_gk = team2_gk.rename(index={len(team2_gk)-1: max(teams, key=teams.get)})
    team2_gk = team2_gk.iloc[-1:]
    #team2_gk

    team2_dfs = [team2_stand, team2_pass, team2_def, team2_pos, team2_misc, team2_gk]
    team2_merge = reduce(lambda  left, right: pd.merge(left, right, left_index=True, right_index=True), team2_dfs)
    team2_merge = team2_merge.apply(pd.to_numeric)
    #team2_merge

    if (team_merge['Goals'][0] > team2_merge['Goals'][0]) == True:
        team_merge['Result'] = 3
        team2_merge['Result'] = 0
    elif (team_merge['Goals'][0] == team2_merge['Goals'][0]) == True:
        team_merge['Result'] = 1
        team2_merge['Result'] = 1
    elif (team_merge['Goals'][0] < team2_merge['Goals'][0]) == True:
        team_merge['Result'] = 0
        team2_merge['Result'] = 3;
    #team_merge
    #team2_merge

    df1_name = (min(teams, key=teams.get) + link.split(max(teams, key=teams.get), 1)[1]).replace("-", "_")
    team_merge = team_merge.rename(index={min(teams, key=teams.get): df1_name})
    #team_merge

    df2_name = (max(teams, key=teams.get) + link.split(max(teams, key=teams.get), 1)[1]).replace("-", "_")
    team2_merge = team2_merge.rename(index={max(teams, key=teams.get): df2_name})
    #team2_merge

    #ca19_game1 = pd.concat([team_merge, team2_merge], axis=0)
    time.sleep(60)
    global e21
    e21 = pd.concat([e21, team_merge, team2_merge], axis=0)
    #return (team_merge, team2_merge)


for i in range(0, len(e21_games)):
    e21_match_data(e21_games[i]);

e21.to_csv("data_tables\euro21.csv", index=True)