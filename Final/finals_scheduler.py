#! /usr/bin/env python
'''
# Team ID:          VD_2537
# Theme:            Vitarana Drone
# Author List:      Jai Kesav, Aswin Sreekumar, Greeshwar R.S, K.Girish
# Filename:         Task_6_VD_2537_scheduling.py,

# Functions:        __init__(),read_from_csv(),write_seq_manifest(),get_cost_table,get_pairs,get_best_pairs
                    multiple_combo_min_val,get_min_combo,Scheduler,Combinator,get_profit_n_cost,knapsack
                    sort_order_key,sort_singles_order_key,sort_rest_key

# Global variables:  delivery_list,delivery_name,return_list,return_name,delivery_rows,return_rows,meter_conv
                     delivery_hoarde,return_hoarde,delivery_index,return_index,cost_table,unordered_cost_table
                     best_pairs,time_delay,time_const,all_pairs_wnv,all_drops,all_pickups,index_list,singles,
                     singles_order,pref_order,tent_combos,first_pref_list,passing_list_rec,Weight,final_schedule

'''






import csv
import math
import itertools
import copy
import rospy

class Scheduling:

    def __init__(self):

        # necessary variables

        rospy.init_node('Scheduler_node', anonymous=True)

        self.read_loc = '/home/jk56/catkin_ws/src/vitarana_drone/scripts/finals.csv'
        self.write_loc = "/home/jk56/catkin_ws/src/vitarana_drone/scripts/manifest.csv"

        # stores locations(pickup and drop) of given set of deliveries in the given order
        self.delivery_list = []
        # Saves grid ID associated with our package for every delivery
        self.delivery_name = []
        # stores locations(pickup and drop) of given set of return in the given order
        self.return_list = []
        # Saves grid ID associated with our package for every return
        self.return_name = []
        # Stores the exact row read from manifest csv for deliveries and returns respectively
        self.delivery_rows = []
        self.return_rows = []
        # Stores conversion values of lattitude/longitude to meters 
        self.meter_conv = [110692.0702932625, 105292.0089353767]
        # Approximate location of the warehouse building
        self.delivery_hoarde = [19, 72, 0]
        self.return_hoarde = [19, 72, 0]
        # incremented every time a delivery/ return is read in manifest. Thus holds the total number
        # returns and deliveries accordingly
        self.delivery_index = 0
        self.return_index = 0

        # stores the return package index and a list of delivery indices, ordered by proximity
        self.cost_table = []
        # stores the return package index and a list of delivery indices, ordered by index value
        self.unordered_cost_table = []
        # stores the best return-delivery pairs that will be computed in the code
        self.best_pairs = []
        # some parameters to calculate the time cost of each service
        self.time_delay = 1                # random value
        self.time_const = 1
        # stores the cost and profit associated with each of the pairs in self.best_pairs
        self.all_pairs_wnv = []
        # stores the cost and profit associated with each individual deliveries
        self.all_drops = []
         # stores the cost and profit associated with each individual returns
        self.all_pickups = []

        self.index_list = []
        # dictionary storing the values that are involved in the 0-1 knapsack to reduce time complexity
        self.t = dict()
        # stores all single return/ delivery services
        self.singles = []
        # stores all singe return/ delivery services in order
        self.singles_order = []

        # used in self.multiple_combo_min_val function

        # list with order of preference of deliveries for each return index
        self.pref_order = []      
        # list storing tentative combinations of return- delivery pairs that dont overlap  
        self.tent_combos = []       
        # first passed_list value for self.multiple_combo_min_val function
        # contains the first preference for each return index 
        self.first_pref_list = []   
        # maintains a record of passed passed_list arguments to reduce time complexity
        self.passing_list_rec = []

        # maximum weight allowed for the 1-0 knapsack algo
        self.Weight = 5000
        # stores the final schedule 
        self.final_schedule = []

        self.read_from_csv()
        self.get_cost_table()
        self.get_pairs()
        self.Combinator()
        self.Scheduler()
        self.write_seq_manifest()

        # print(self.best_pairs)

    # ---------------Functions--------------------

    # Reading from CSV file
    def read_from_csv(self):

        '''
        Purpose:
        ---
        reads from manifest file and saves the readings in self.return_list and self.return_row
        < Short-text describing the purpose of this function >  
        ---
        NONE

        Example call:
        ---
        read_from_csv()
        '''

        with open(self.read_loc) as manifest_csv:
            csv_reader = csv.reader(manifest_csv, delimiter=',')

            for row in csv_reader:
                if row[0] == 'DELIVERY':
                    coord = str(row[2]).split(';')
                    coord = [float(coord[0]), float(coord[1]), float(coord[2])]
                    self.delivery_list.append([self.delivery_index, coord])
                    self.delivery_name.append(row[1])
                    self.delivery_rows.append(row)
                    self.delivery_index += 1

                elif row[0] == 'RETURN':
                    coord = str(row[1]).split(';')
                    coord = [float(coord[0]), float(coord[1]), float(coord[2])]
                    self.return_list.append([self.return_index, coord])
                    self.return_name.append(row[2][0:2])
                    self.return_rows.append(row)
                    self.return_index += 1

    def write_seq_manifest(self):
        '''
        Purpose:
        ---
        writes to sequenced_manifest_original.csv from data of self.final_schedule        

        Input Arguments:
        ---
        NONE

        Returns:
        ---
        NONE

        Example call:
        ---
        write_seq_manifest()
'''

        with open(self.write_loc, mode='w') as csv_file:
            csv_writer = csv.writer(csv_file, dialect='excel')
            for i in self.final_schedule:
                if i[0] == "pick - return":
                    csv_writer.writerow(self.return_rows[i[1]])
                elif i[0] == "pick - delivery":
                    csv_writer.writerow(self.delivery_rows[i[1]])

        print("----- Sequence Manifest Generated -------")


    # Getting distance of pickup from each delivery location
    def get_cost_table(self):
        '''
        Purpose:
        ---
        from the location data in self.delivery_list and self.return_list, this function calculates the distance
        between each return and delivery location and stores it in self.cost_table and self.unordered_cost_table.
        The latter 2 variables will be used to decide the closest marker/ delivery location to for each return 
        package pickup position

        Input Arguments:
        ---
        NONE

        Returns:
        ---
        NONE
        Example call:
        ---
        get_cost_table()
        '''


        for i in range(self.return_index):
            ret_loc = self.return_list[i][1]
            cost_list = []
            for j in range(self.delivery_index):
                del_loc = self.delivery_list[j][1]
                lat_diff = (ret_loc[0] - del_loc[0])*self.meter_conv[0]
                lon_diff = (ret_loc[1] - del_loc[1])*self.meter_conv[1]
                linear_diff = math.sqrt(lat_diff**2 + lon_diff**2)
                cost_list.append((linear_diff, j))

            self.unordered_cost_table.append([i, cost_list])
            cost_list = sorted(cost_list)
            self.cost_table.append([i, cost_list])

        print("-- ordered CT --")
        for i in self.cost_table:
            print(i)


    # get the pickup - delivery pairs
    def get_pairs(self):
        '''
        Purpose:
        ---
        Tries to get the best possible return - delivery pairs based on self.cost_table. Finds if multiple return package
        location has the same closest delivery location, if yes, it calls self.get_best_pairs and passes undec_ret and
        rep_dels (undecided returns and repeated deliveries)
        

        Input Arguments:
        ---
        NONE

        Returns:
        ---
        NONE

        Example call:
        ---
        get_pairs()
        '''

        

        pairs = []
        del_order = []  # has the order of closest delivery corresponding to each return
        for i in range(len(self.cost_table)):
            # return idx, delivery idx
            pairs.append([self.cost_table[i][0], self.cost_table[i][1][0][1]])
            del_order.append(self.cost_table[i][1][0][1])

        rep_dels = []  # repeated deliveries in pairs
        for i in range(self.return_index):
            if del_order.count(i) != 1:
                rep_dels.append(i)

        undec_ret = []  # for what all returns I should optimize delivery-return pair
        for i in pairs:
            if i[1] in rep_dels:
                undec_ret.append(i[0])

        # min_val_pairs = self.get_best_pairs(undec_ret, rep_dels)[1]
        # for i in range(len(undec_ret)):
        #     pairs[undec_ret[i]] = [undec_ret[i], min_val_pairs[i]]

        # self.best_pairs = pairs
        self.get_best_pairs(undec_ret, rep_dels)
        print("best pairs : ", self.best_pairs)

    # Used to get the least cost pairing for undecided returns
    def get_best_pairs(self, undecided_returns, rep_dels):
        '''
        Purpose:
        ---
        Searches if a simple combination will optimize the return package - delivery location pairs. If not a simple case,
        it updates variables required for rigorous/perfect scheduling and calls self.multiple_combo_min_val to do the same.
        In the latter case, self.multiple_combos_in_val will be run with incremented max_idx values until a set of tentative
        combinations of non overlapping return package - delivery pairs
        Updates self.best_pairs with the optimal return package - delivery location pairs

        Input Arguments:
        ---
        `undecided_returns` : list 
            list of return packages that dont have the perfect delivery pair
            

        `rep_dels` :  list 
            list of delivery pairs that dont have the perfect return package
            

        Returns:
        ---

        Example call:
        ---
        < Example of how to call this function >
        '''

        l = len(undecided_returns)
        if l == 0:
            pass

        elif l == 2:
            r1_opt = [self.cost_table[undecided_returns[0]][1][0][1], self.cost_table[undecided_returns[0]][1][1][1]]  # closest 2 deliveries
            r2_opt = [self.cost_table[undecided_returns[1]][1][0][1], self.cost_table[undecided_returns[1]][1][1][1]]  # closest 2 deliveries
            if r1_opt == r2_opt:
                min_values = self.get_min_combo(undecided_returns, r1_opt)
                return min_values


        else:
            # Get the best pairs by using the cost table
            # Kinda complicated, but have fun

            # self.pref_order = []        # X - the list with order of preference
            # self.tent_combos = []       # combos - to get the final possibilities
            # self.first_pref_list = []   # initial list with all the first prefs

            for i in self.cost_table:
                l = []
                self.first_pref_list.append(i[1][0][1])
                for j in i[1]:
                    l.append(j[1])
                self.pref_order.append(l)

            print("len pref list : ", len(self.pref_order))
            print("pref list ----")
            for i in self.pref_order:
                print(i)

            print("first pref list : ", self.first_pref_list)

            self.tent_combos = []
            for i in range(len(self.cost_table)):
                print("TC : ", self.tent_combos)
                self.passing_list_rec = []
                if len(self.tent_combos) != 0:
                    break
                else:
                    print("max_idx : ", i)
                    self.multiple_combo_min_val(i, self.first_pref_list)

            print("---- tentative combos ----- ")
            for i in self.tent_combos:
                print(i)

            min = float('inf')
            combo_idx = -1
            for i in self.tent_combos:
                cost = 0
                for j in range(9):
                    ret_idx = j
                    del_idx = i[j]
                    cost += self.unordered_cost_table[ret_idx][1][del_idx][0]

                if cost <= min:
                    combo_idx = self.tent_combos.index(i)
                    min = copy.copy(cost)
                    print("going low : ", min, combo_idx)

            min_vals = []
            ret_idx = 0
            for i in self.tent_combos[combo_idx]:
                min_vals.append([ret_idx, i])
                ret_idx +=1

            print("min_vals : ")
            for i in min_vals:
                print(i)

            self.best_pairs = min_vals




    def multiple_combo_min_val(self, max_idx, passed_list, p=0):
        '''
        Purpose:
        ---
        Based on the delivery location preference of each return that can be found in self.cost_table, it tries to find combinations
        of return package - delivery pair that does'nt overlap. Its a recursive function, explaining its working is beyond the
        scope of comment section
        Input Arguments:
        ---
        `max_idx` : int [ < type of 1st input argument > ]
            gets the maximum preference number that can be taken for each retrun package 
            < one-line description of 1st input argument >

        `passed list` : list [ < type of 2nd input argument > ]
            The list containing a series of delivery index, assuming the list index to be the index of the return package.
            < one-line description of 2nd input argument >

        Returns:
        ---
        < name of 1st return argument >` :  [ < type of 1st return argument > ]
            < one-line description of 1st return argument >

        < name of 2nd return argument >` :  [ < type of 2nd return argument > ]
            < one-line description of 2nd return argument >

        Example call:
        ---
        self.multiple_combo_min_val(max_idx, passed_list)
        < Example of how to call this function >
        '''

        repeats = []
        if passed_list in self.passing_list_rec:
            pass
        else:
            self.passing_list_rec.append(passed_list)
            for i in passed_list:
                count = passed_list.count(i)
                if count > 1:
                    repeats.append([i, count])

            if len(repeats) != 0:

                for i in repeats:
                    idx_count = 0
                    rep_v_idx = - 1
                    while idx_count != i[1]:

                        rep_v_idx = passed_list[rep_v_idx + 1:].index(i[0]) + rep_v_idx + 1
                        idx_count += 1
                        list_idx = self.pref_order[rep_v_idx].index(i[0]) + 1

                        if list_idx <= max_idx:
                            temp_list = list(passed_list)
                            temp_list[rep_v_idx] = self.pref_order[rep_v_idx][list_idx]
                            self.multiple_combo_min_val(max_idx, temp_list, p+1)
                            # print("passing : ", temp_list, p, max_idx)
                            # print("actual : ", passed_list)


            else:
                if passed_list not in self.tent_combos:
                    self.tent_combos.append(passed_list)



    # Gets the minimum cost for all the possible combinations
    def get_min_combo(self, undecided_returns, del_options):
        '''
        Purpose:
        Same as self.multiple_combo_min_val(), but for less complex cases
        ---

        Input Arguments:
        ---
        `undecided_returns` : list 
            list of return packages that dont have the perfect delivery pair            

        `rep_dels` :  list 
            list of delivery pairs that dont have the perfect return package

        Returns:
        ---
        'min_values' :  list
            list of certain return package - delivery pair that is obtained by processing the passed arguments

        Example call:
        min_values = self.get_min_combo(undec_rets, ret_dels)
        ---
        
        '''


        cond_list = []
        min_values = [float('inf'), [0]]
        for i in undecided_returns:
            pos_list = []
            for j in range(self.delivery_index):
                if self.cost_table[i][1][j][1] in del_options:
                    pos_list.append(self.cost_table[i][1][j][1])
            cond_list.append(pos_list)

        for combination in itertools.product(*cond_list):
            if len(combination) == len(set(combination)):
                cost = 0
                for i in range(len(combination)):
                    cost += self.unordered_cost_table[undecided_returns[i]][1][combination[i]][0]
                if cost < min_values[0]:
                    min_values[0] = cost
                    min_values[1] = combination

        return min_values

    # Schedules based on maximum profit in limited cost(time)
    def Scheduler(self):
        '''
        Purpose:
        ---
            Runs the 0 - 1 knapsack algo to schedule to the return package - delivery pairs along with some singular returns 
            or deliveriers if demanded by the Weight situation, and updates it in self.final_schedule        

        Input Arguments:
        ---
        NONE

        Returns:
        ---
        NONE

        Example call:
        ---
        Scheduler()
        '''

        # cleverly combs through [cost, profit] of every combination and decides
        # maybe try having profit/cost ratio as a parameter
        wt = [i[1][0] for i in self.all_pairs_wnv]
        val = [i[1][1] for i in self.all_pairs_wnv]
        order = self.knapsack(wt, val, self.Weight, len(self.all_pairs_wnv))[1]
        print("order : ", order)
        order = sorted(order, key=self.sort_order_key)
        print("orderded order : ", order)
        weight_used = 0
        done_pickups = []
        done_drops = []
        for i in order:
            # appending in final schedule
            delivery_idx = self.all_pairs_wnv[i-1][0][1]
            return_idx = self.all_pairs_wnv[i-1][0][0]
            self.final_schedule.append(("pick - delivery", delivery_idx, self.delivery_name[delivery_idx]))
            self.final_schedule.append(("drop - delivery", delivery_idx, self.delivery_list[delivery_idx][1]))
            self.final_schedule.append(("pick - return", return_idx, self.return_list[return_idx][1]))
            self.final_schedule.append(("drop - return", return_idx, self.return_name[return_idx]))

            weight_used += self.all_pairs_wnv[i-1][1][0]
            done_pickups.append(self.all_pairs_wnv[i-1][0][0])
            done_drops.append(self.all_pairs_wnv[i-1][0][1])

        weight_remaining = self.Weight - weight_used
        self.singles = []
        for i in range(self.delivery_index):
            if i not in done_drops:
                temp = self.all_drops[i]
                self.singles.append([temp[0], [temp[1][0]/2, temp[1][1], temp[1][2]]])

        for i in range(self.return_index):
            if i not in done_pickups:
                temp = self.all_pickups[i]
                self.singles.append(temp)

        wt = [i[1][0] for i in self.singles]
        val = [i[1][1] for i in self.singles]

        self.t = dict()
        self.index_list = []
        self.singles_order = self.knapsack(wt, val, weight_remaining, len(self.singles))[1]
        self.singles_order = sorted(self.singles_order, key=self.sort_singles_order_key)

        for i in self.singles_order:
            # appending in schedule
            delivery_idx = self.singles[i-1][0][1]
            return_idx = self.singles[i-1][0][0]
            if delivery_idx == -1:
                # its a return
                self.final_schedule.append(("pick - return", return_idx, self.return_list[return_idx][1]))
                self.final_schedule.append(("drop - return", return_idx, self.return_name[return_idx]))
                done_pickups.append(return_idx)
            else:
                # its a delivery
                self.final_schedule.append(("pick - delivery", delivery_idx,  self.delivery_name[delivery_idx]))
                self.final_schedule.append(("drop - delivery", delivery_idx, self.delivery_list[delivery_idx][1]))
                done_drops.append(delivery_idx)

        rest = []
        for i in self.singles:
            if i[0][0] == -1 and (i[0][1] not in done_drops):
                # its a delivery
                rest.append(i)
            elif i[0][1] == -1 and (i[0][0] not in done_pickups):
                # its a return
                rest.append(i)

        rest = sorted(rest, key=self.sort_rest_key)

        for i in rest:
            delivery_idx = i[0][1]
            return_idx= i[0][0]

            if delivery_idx == -1:
                # its a return
                self.final_schedule.append(("pick - return", return_idx,  self.return_list[return_idx][1]))
                self.final_schedule.append(("drop - return", return_idx,  self.return_name[return_idx]))
                done_pickups.append(return_idx)
            else:
                # its a delivery
                self.final_schedule.append(("pick - delivery", delivery_idx, self.delivery_name[delivery_idx]))
                self.final_schedule.append(("drop - delivery", delivery_idx, self.delivery_list[delivery_idx][1]))
                done_drops.append(delivery_idx)


        print("_____ Final Schedule ________")
        for i in self.final_schedule:
            print(i)

        print(len(self.final_schedule))



    # Gets all possible combos from self.best_pairs
    def Combinator(self):
        '''
        Purpose:
        ---
        Gets the cost(for whole journey) and profit for each of the pairs in self.best_pairs and for individual returns and
        individual deliveries. This data will be stored in self.all_pairs_wnv, self.all_pickups and self.all_drops respectively.
        The profit and cost will be obtained by calling the self.get_profit_n_cost(pair) function for each pair 
        < Short-text describing the purpose of this function >

        Input Arguments:
        ---
        NONE

        Returns:
        ---
        NONE
        Example call:
        ---
        self.Combinator()
        '''

        # using delivery hoarde and return hoarde locations, calculate cost and profit
        # for every combination in self.best_pairs. Also for individual returns and deliveries

        for i in self.best_pairs:
            val = self.get_profit_n_cost(i)
            self.all_pairs_wnv.append([i, val])

        for i in self.delivery_list:
            val = self.get_profit_n_cost([-1, i[0]])
            self.all_drops.append([[-1, i[0]], val])

        for i in self.return_list:
            val = self.get_profit_n_cost([i[0], -1])
            self.all_pickups.append([[i[0], -1], val])

        for i in self.all_pairs_wnv:
            print(i)
        for i in self.all_drops:
            print(i)
        for i in self.all_pickups:
            print(i)


    # Get profit, cost and profit/cost ratio of passed pair
    # in the form [cost, profit, profit/cost]
    def get_profit_n_cost(self, pair):
        '''
        Purpose:
        ---
        returns the profit and cost of the whole journey for each of the passed pair argument.

        Input Arguments:
        ---
        `pair` :  list
            [return index, delivery index]

        

        Returns:
        ---
        [cost, profit, profit/cost] : list
            self explanatory        

        Example call:
        ---
        val = self.get_profit_n_cost([i[0], -1])
        '''


        loc_1 = self.delivery_list[pair[1]][1]
        loc_2 = self.return_list[pair[0]][1]

        if pair[0] == -1:
            # linear distance
            lat_diff = (loc_1[0] - self.delivery_hoarde[0]) * self.meter_conv[0]
            lon_diff = (loc_1[1] - self.delivery_hoarde[1]) * self.meter_conv[1]
            linear_dist = math.sqrt(lat_diff ** 2 + lon_diff ** 2)

            cost = 2*linear_dist*self.time_const + 2*self.time_delay
            profit = 5 + 0.1*linear_dist

        elif pair[1] == -1:
            # linear distance
            lat_diff = (loc_2[0] - self.return_hoarde[0]) * self.meter_conv[0]
            lon_diff = (loc_2[1] - self.return_hoarde[1]) * self.meter_conv[1]
            linear_dist = math.sqrt(lat_diff ** 2 + lon_diff ** 2)

            cost = 2 * linear_dist*self.time_const + 2 * self.time_delay
            profit = 5 + 0.1 * linear_dist

        else:
            lat_diff1 = (loc_1[0] - self.delivery_hoarde[0]) * self.meter_conv[0]
            lon_diff1 = (loc_1[1] - self.delivery_hoarde[1]) * self.meter_conv[1]
            linear_dist1 = math.sqrt(lat_diff1 ** 2 + lon_diff1 ** 2)

            lat_diff2 = (loc_2[0] - self.return_hoarde[0]) * self.meter_conv[0]
            lon_diff2 = (loc_2[1] - self.return_hoarde[1]) * self.meter_conv[1]
            linear_dist2 = math.sqrt(lat_diff2 ** 2 + lon_diff2 ** 2)

            linear_dist = linear_dist1 + linear_dist2 + self.unordered_cost_table[pair[0]][1][pair[1]][0]

            cost = linear_dist*self.time_const + 3*self.time_const
            profit = 10 + 0.1*linear_dist

        return [cost, profit, profit/cost]

    def knapsack(self, wt, val, W, n):
        '''
        Purpose:
        ---
        based on the cost of each service and the total available weight, it decides what 
        services to do to optimise the max profit in given time. A modified 0-1 knapsack algo 
        is implemented - recursive in nature

        Input Arguments:
        ---
        `wt` :  list
            list of cost of each service

        `val` :  list
            list of profit of each pair

        `W` :  float/ int
           Total available cost, based on time

        `n` :  int
            length of wt or val list



        Returns:
        ---
        `self.t[key]` :  list [profit, [service indices]]
            gets profit assiociated with taking the services represented by the service indices
        
        Example call:
        ---
        order = self.knapsack(wt, val, self.Weight, len(self.all_pairs_wnv))
        '''


        key = str(n) + "," + str(W)
        # base conditions
        if n == 0 or W == 0:
            return [0, []]

        if key in self.t:
            return self.t[key]

        # choice diagram code
        if wt[n - 1] <= W:

            knap1 = self.knapsack(wt, val, W - wt[n - 1], n - 1)
            knap2 = self.knapsack(wt, val, W, n - 1)

            if val[n - 1] + knap1[0] >= knap2[0]:
                knap1[1].append(n)
                self.t[key] = [val[n - 1] + knap1[0], knap1[1]]
                for i in knap2[1]:
                    if i in self.index_list:
                        self.index_list.remove(i)
                for i in knap1[1]:
                    if i not in self.index_list:
                        self.index_list.append(i)

            else:
                self.t[key] = knap2
                for i in knap1[1]:
                    if i in self.index_list:
                        self.index_list.remove(i)
                for i in knap2[1]:
                    if i not in self.index_list:
                        self.index_list.append(i)

            return self.t[key]

        elif wt[n - 1] > W:
            self.t[key] = self.knapsack(wt, val, W, n - 1)
            return self.t[key]

    def sort_order_key(self, i):
        '''
        Purpose:
        ---
        returns a value in self.all_pairs_wnv for sorted() as key

        Input Arguments:
        ---
        `i` :  int
            specifices the index in self.all_pairs_wnv

        
        Returns:
        ---
        self.all_pairs_wnv[i-1][1][2] :  float/ int
            cost or profit associated

        

        Example call:
        ---
        order = sorted(list, k=sort_order_key)
        '''

        return self.all_pairs_wnv[i-1][1][2]

    def sort_singles_order_key(self, i):
        '''
        Purpose:
        ---
        returns a value in self.singles for sorted() as key

        Input Arguments:
        ---
        `i` :  int
            specifices the index in self.singles

        
        Returns:
        ---
        self.singles[i-1][1][0] :  float/ int
            cost or profit associated

        

        Example call:
        ---
        order = sorted(list, k=sort_singles_order_key)
        '''

        return self.singles[i-1][1][0]

    def sort_rest_key(self, i):
        '''
        Purpose:
        ---
        returns a value in passed list for sorted() as key

        Input Arguments:
        ---
        `i` :  int
            specifices the index in passed list

        
        Returns:
        ---
        i[1][0] :  float/ int
            value associated        

        Example call:
        ---
        rest = sorted(rest, key=self.sort_rest_key)
        '''
        return i[1][0]

if __name__ == "__main__":

    schedule = Scheduling()