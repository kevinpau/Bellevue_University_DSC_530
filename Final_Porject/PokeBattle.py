#%%
"""
This module contains a class to simulate a simplified pokemon battle.

Assumptions:
1) challenger pokemon will have full health when facing a new elite 4 trainer
2) only attack and defense will be used not sp_attack or sp_defense
3) damange done to hp is simplified (attack - defense), if neg dif only deal 10% of diff
"""

import pandas as pd
import numpy as np
import random

class PokeBattle1():
    """ this class simulates pokemon battle

    """
    # elite 4 pokemon
    elite4 = {'Lorelei':['Dewgong','Cloyster','Slowbro','Jynx','Lapras'],
    'Bruno':['Onix','Hitmonchan','Hitmonlee','Onix','Machamp'],
    'Agatha':['Gengar','Golbat','Haunter','Arbok','Gengar'],
    'Lance':['Gyarados','Dragonair','Dragonair','Aerodactyl','Dragonite']}

    # we want to fight in order so a list of trainers is needed
    #elite4Trainers = ['Lorelei', 'Bruno', 'Agatha', 'Lance']
    elite4Trainers = ['Lance']

    def __init__(self, pokemon):
        """ initalizes the class

        @param: pokemon (list) - list of pokemon
        """
        self.pokemon = pokemon

        self.df = self.readDataFrame()
        self.df_challenger = pd.DataFrame()
        self.df_elite4 = pd.DataFrame()

    def readDataFrame(self):
        """ function reads/cleans dataframe of pokemon data
        """
        df_all = pd.read_csv('pokemon.csv')

        cols = "name hp	attack defense speed base_total type1 type2 generation".split()

        # get onely gen 1 pokemon
        df_gen1 = df_all[cols].loc[df_all['generation'] == 1]

        return df_gen1

    def createChallenger(self):
        """ function to set up/reset challenger pokemon
        """
        # find pokemon and add to new df
        for p in self.pokemon:
            self.df_challenger = self.df_challenger.append(
                self.df.loc[self.df["name"] == p])

    def createElite4(self):
        """ function to set up/reset elite 4 pokemon
        """
        # list of elite 4 pokemon
        poke = [v for key, val in PokeBattle1.elite4.items() for v in val]

        for p in poke:
            self.df_elite4 = self.df_elite4.append(self.df.loc[self.df["name"] == p])

    def startBattle(self):
        """ function to initiate battle
        """
        self.createElite4()

        challengerStatus = ""

        for trainer in PokeBattle1.elite4Trainers:
            if challengerStatus == "lost":
                break

            self.createChallenger() # assume the pokemon are at full health

            elite4Poke = PokeBattle1.elite4[trainer]

            # select pokemon and start fight
            count = 0
            for p in self.pokemon:
                count += 1
                hp = self.df_challenger.loc[(self.df_challenger["name"] == p)].iloc[0]["hp"]
                
                if hp > 0:

                    ecount = 0
                    for ep in elite4Poke:
                        ecount += 1
                        hp2 = self.df_elite4.loc[(self.df_elite4["name"] == ep)].iloc[0]["hp"]

                        if hp2 > 0:
                            print("\n")
                            print("Challenger: {} vs. {}: {}".format(p, trainer, ep))
                            status = self.fightElite(p, ep)

                            if status == 0:
                                break # challenger lost need next p
                            
                            if status == 1 and ecount == len(elite4Poke):
                                print("\nTrainer {} is out of pokemon".format(trainer))
                   
                hp = self.df_challenger.loc[(self.df_challenger["name"] == p)].iloc[0]["hp"]

                if hp <= 0 and count == len(self.pokemon):
                    print("\nChallenger is out of pokemon, you need a new lineup")
                    challengerStatus = "lost"
                    
    def fightElite(self, chalPoke, elitePoke):
        """ function fights two pokemon

        @param: chalPokem (str) - name of challenger pokemon
        @param: elitePoke (Str) - name of elite4 pokemon
        """
        # get values of each pokemon
        d = self.df_challenger.loc[self.df_challenger["name"] == chalPoke]
        chal_index = d.index[0]
        chal_hp = d.iloc[0]["hp"]
        chal_attack = d.iloc[0]["attack"]
        chal_defense = d.iloc[0]["defense"]

        d = self.df_elite4.loc[self.df_elite4["name"] == elitePoke]
        elite_index = d.index[0]
        elite_hp = d.iloc[0]["hp"]
        elite_attack = d.iloc[0]["attack"]
        elite_defense = d.iloc[0]["defense"]

        print("{}: {}hp vs {}: {}hp".format(chalPoke, chal_hp, elitePoke, elite_hp))

        while(chal_hp > 0 and elite_hp > 0):
            
            if chal_hp > 0:
                # chal attack elite
                diff = chal_attack - elite_defense

                # if attack and defense =, take away 10% next turn
                if diff == 0:
                    elite_defense -= elite_defense*0.9

                if diff > 0:
                    #print("{} deals {} damange to {}".format(chalPoke, diff, elitePoke))
                    elite_hp -= diff

                if diff < 0:
                    #print("{} attacked but it was not very effective".format(chalPoke))
                    elite_hp -= (abs(diff) * 0.1)

            if elite_hp > 0:
                # elite attack chal
                diff = elite_attack - chal_defense

                if diff == 0:
                    chal_defense -= chal_defense*0.9

                if diff > 0:
                    #print("{} deals {} damange to {}".format(elitePoke, diff, chalPoke))
                    chal_hp -= diff
                
                if diff < 0:
                    #print("{} attacked but it was not very effective".format(elitePoke))
                    chal_hp -= (abs(diff) * 0.1)

        # update the dataframes
        if chal_hp < 0 or chal_hp == 0:
            chal_hp = 0
            print("{} fainted".format(chalPoke))
            print("{} is the winner".format(elitePoke))    
            
        
        if elite_hp < 0 or elite_hp == 0:
            elite_hp = 0
            print("{} fainted".format(elitePoke))
            print("{} is the winner".format(chalPoke))            

        self.df_challenger.at[chal_index, "hp"] = chal_hp
        self.df_elite4.at[elite_index, "hp"] = elite_hp

        # report the status and move on to next fight
        if chal_hp <= 0:
            # challenger lost
            return 0
        else:
            # challenger won
            return 1
