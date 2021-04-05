import csv
import math
import itertools

class Scheduling:

    def __init__(self):

        # necessary variables

        self.delivery_list = []
        self.delivery_name = []
        self.return_list = []
        self.return_name = []
        self.delivery_rows = []
        self.return_rows = []
        self.meter_conv = [110692.0702932625, 105292.0089353767]
        self.delivery_hoarde = [19, 72, 0]
        self.return_hoarde = [19, 72, 0]
        self.delivery_index = 0
        self.return_index = 0

        self.cost_table = []
        self.unordered_cost_table = []
        self.best_pairs = []
        self.time_delay = 1                # random value
        self.time_const = 1
        self.all_pairs_wnv = []
        self.all_drops = []
        self.all_pickups = []
        self.index_list = []
        self.t = dict()
        self.singles = []
        self.singles_order = []

        self.Weight = 1500
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
        with open('manifest.csv') as manifest_csv:
            csv_reader = csv.reader(manifest_csv, delimiter=',')

            for row in csv_reader:
                if row[0] == 'DELIVERY':
                    coord = str(row[2]).split(';')
                    coord = [float(coord[0]), float(coord[1]), float(coord[2])]
                    self.delivery_list.append([self.delivery_index, coord])
                    self.delivery_name.append(row[1])
                    self.delivery_rows.append(row)
                    self.delivery_index += 1

                elif row[0] == 'RETURN ':
                    coord = str(row[1]).split(';')
                    coord = [float(coord[0]), float(coord[1]), float(coord[2])]
                    self.return_list.append([self.return_index, coord])
                    self.return_name.append(row[2][0:2])
                    self.return_rows.append(row)
                    self.return_index += 1

    def write_seq_manifest(self):
        with open("sequenced_manifest.csv", mode='w', newline='') as csv_file:
            csv_writer = csv.writer(csv_file, dialect='excel')
            for i in self.final_schedule:
                if i[0] == "pick - return":
                    csv_writer.writerow(self.return_rows[i[1]])
                elif i[0] == "pick - delivery":
                    csv_writer.writerow(self.delivery_rows[i[1]])

        print("----- Sequence Manifest Generated -------")


    # Getting distance of pickup from each delivery location
    def get_cost_table(self):
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


    # get the pickup - delivery pairs
    def get_pairs(self):
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

        min_val_pairs = self.get_best_pairs(undec_ret, rep_dels)[1]
        for i in range(len(undec_ret)):
            pairs[undec_ret[i]] = [undec_ret[i], min_val_pairs[i]]

        self.best_pairs = pairs

    # Used to get the least cost pairing for undecided returns
    def get_best_pairs(self, undecided_returns, rep_dels):
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

            pass

    # Gets the minimum cost for all the possible combinations
    def get_min_combo(self, undecided_returns, del_options):

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
        return self.all_pairs_wnv[i-1][1][2]

    def sort_singles_order_key(self, i):
        return self.singles[i-1][1][0]

    def sort_rest_key(self, i):
        return i[1][0]

if __name__ == "__main__":

    schedule = Scheduling()