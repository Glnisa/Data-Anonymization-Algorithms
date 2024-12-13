import copy

##some of them are not used but implemented once and I do not have time to find and delete them... :(
###########################################
##NODE - ROOT
###########################################
class Node:

    def __init__(self, name, depth, parent=None, child=None):
        self.name = name
        self.depth = depth
        self.parent = parent
        self.child = list()
        self.count = 0
        if child is not None:
            self.child.append(child)

    def count_leaves(self):
        leaves = []
        self.helper_func(leaves)
        return len(leaves)

    def helper_func(self, leaves):
        if not self.child:
            leaves.append(self)
        else:
            for c in self.child:
                c.helper_func(leaves)

    def traverse(self, height_count, max_depth):

        current_node = self

        if height_count == 0:
            return current_node
        depth = max_depth - height_count

        if current_node.depth <= depth:
            return current_node

        while True:
            if current_node.parent != None:
                current_node = current_node.parent

                if current_node.name == "Any":
                    return current_node

                if current_node.parent.depth == depth:
                    return current_node



class Root:

    def __init__(self, domain_name, child=None):
        self.name = domain_name
        self.child = child
        self.max_depth = 0

    def elevator(self, depth_tree):
        if depth_tree == 1:
            return self.child
        else:
            last_node = self.child
            for i in range(depth_tree - 1):
                last_node = last_node.child[-1]

            return last_node


###########################################
##BOTTOM-UP HELPERS
###########################################


def max_depth(root) -> int:
    if root:
        child_depths = list(map(max_depth, root.child))
        return max(child_depths, default=0) + 1

    else:
        return 0  #indicates no child

def change_charac(string, position, char):
    string = string[:position] + str(char) + string[position + 1:]
    return string

def dfs_in_DGH(root, node_name):

    if root.name == node_name:
        return root

    stack, path = [root], []

    while stack:

        v = stack.pop()
        if v in path:
            continue
        path.append(v)

        for neighbour in v.child:

            if neighbour.name == node_name:
                return neighbour
            
            stack.append(neighbour)
            


def find_max_h_DGHs(DGHs):
    max_generalization = 1

    for d in DGHs.values():

        d.max_depth = max_depth(d.child)
        max_generalization = max_generalization * d.max_depth

    return max_generalization



def k_anonymity_check(DGHs, anonymized_dataset, k):
    counter = 0

    for r in anonymized_dataset:

        if r["check"]:
            continue

        r["check"] = True

        for raw_done in anonymized_dataset:
            is_equal = True

            for domain in DGHs.values():
                if raw_done[domain.name] != r[domain.name]:
                    is_equal = False

            if is_equal:
                counter += 1

        if counter >= (k-1):
            counter = 0
            continue

        else:
            return False

    return True
def l_diversity_check(DGHs, anonymized_dataset, l):


    if not anonymized_dataset:
        raise ValueError("The dataset is empty.")

    
    sensitive_attribute = list(anonymized_dataset[0].keys())[-1]

    
    equivalence_classes = {}
    for record in anonymized_dataset:
        
        key = tuple(record[domain.name] for domain in DGHs.values())
        if key not in equivalence_classes:
            equivalence_classes[key] = []
        equivalence_classes[key].append(record)

    
    for ec in equivalence_classes.values():
        
        sensitive_values = set(record[sensitive_attribute] for record in ec)
    
        if len(sensitive_values) < l:
            return False

    return True


def generalize_record(record, DGHs, current_level):
    for domain in DGHs.values():
        current_node = dfs_in_DGH(domain.child, record[domain.name])
        target_depth = current_node.depth - current_level
        while current_node.parent and current_node.depth > target_depth:
            current_node = current_node.parent
        record[domain.name] = current_node.name
    
###########################################
#RANDOMIZED HELPERS
###########################################

def find_ECs(raw_dataset, k):
    start = 0
    ECs = list()

    for _ in range(0, len(raw_dataset), k):
        k_data = raw_dataset[start:start + k]
        start += k
        ECs.append(k_data)

    return ECs


def generalize_data(DGHs, ECs):

    name_list_domain = list(DGHs.values())

    for d in name_list_domain:

        domain_name = d.name
        domain_dict = dict()

        for EC in ECs:

            nodes = list()
            generalized_list = list()
            
            for data in EC:

                if data[domain_name] in domain_dict:
                    selected = domain_dict[data[domain_name]]

                else:
                    selected = dfs_in_DGH(d.child, data[domain_name])
                    domain_dict[data[domain_name]] = selected

                nodes.append(selected)

            traversed = traverse_for_generalization(nodes)
            generalized_domains = generalize_nodes(traversed)

            data[domain_name] = generalized_domains[0].name
            generalized_list.append(data)

    return ECs


def traverse_for_generalization(nodes):

    depths = list()
    domain_values = list()

    for n in nodes:
        depths.append(n.depth)
        domain_values.append(n.name)

    boolean_equal_depth = all(element == depths[0] for element in depths)

    if not boolean_equal_depth:
        min_depth = min(depths)
        traversed_nodes = depth_level_traversal(nodes, min_depth)
        
    else:
        traversed_nodes = nodes

    return traversed_nodes


def depth_level_traversal(nodes: list, depth: int):
    traversed_nodes = list()

    for n in nodes:

        traversed_node = n

        if n.depth != depth:
            while traversed_node.depth != depth:
                traversed_node = traversed_node.parent

        traversed_nodes.append(traversed_node)

    return traversed_nodes

def generalize_nodes(generalized_nodes):

    while True:
        first = generalized_nodes[0].name 
        boolean_equal_domain = all(e.name == first
                                   for e in generalized_nodes)
        if boolean_equal_domain:
            return generalized_nodes
        
        else:
            newly_generalizeds = list()

            for node in generalized_nodes:
                node = node.parent
                newly_generalizeds.append(node)

            generalized_nodes = newly_generalizeds


###########################################
#CLUSTERING HELPERS
###########################################
def add_check_col(raw_dataset):

    for r in raw_dataset:
        r["check"] = False
    return raw_dataset


def add_cost_col(raw_dataset):

    for r in raw_dataset:
        r["cost"] = 0
    return raw_dataset


def add_index_col(raw_dataset):

    index = 0
    for r in raw_dataset:
        r["index"] = index
        index += 1
    return raw_dataset

def del_meanless_cols(anonymized_dataset):

    for anon_data in anonymized_dataset:
        del anon_data['index']
        del anon_data['check']
        del anon_data['cost']


def find_top_data(anon_ds):

    for a in anon_ds:
        if not anon_ds["check"]:
            return a


def create_generalization_cost_dict(DGHs):

    generalization_cost = dict()
    for d in DGHs.keys():
        generalization_cost[d] = dict()
    return generalization_cost


def calculate_value_of_lm(total_leaves, anonymized_node):

    l=len(anonymized_node.child)
    
    if l == 0:
        numerator = 0
    else:
        numerator = anonymized_node.count_leaves() - 1
    denominator = total_leaves - 1

    value_of_lm = numerator / denominator

    return value_of_lm


def calculate_generalization_cost(DGHs, generalization_cost, anon_d, next_anon_d):

    a = len(DGHs.values())

    for d in DGHs.values():

        selected = anon_d[d.name]
        target = next_anon_d[d.name]

        k = selected + " | " + target

        if k in generalization_cost[d.name]:
            cost_lm = generalization_cost[d.name][k]["cost"]

        else:
            node_list = list()
            total_leaves = d.child.count_leaves()
            node_list.append(dfs_in_DGH(d.child, anon_d[d.name]))
            node_list.append(dfs_in_DGH(d.child, next_anon_d[d.name]))

            traversed_nodes = traverse_for_generalization(node_list)
            generalized_domains = generalize_nodes(traversed_nodes)
            cost_lm = calculate_value_of_lm(total_leaves, generalized_domains[0])

            generalization_cost[d.name][k] = dict()
            generalization_cost[d.name][k]["cost"] = cost_lm

        next_anon_d["cost"] += (cost_lm * (1 / a))

    return next_anon_d


def find_min_generalization_cost(DGhs, customized_dataset: list, k):

    k_min_cost = sorted(customized_dataset, key=lambda d: d['cost'])[:k]
    k_min_cost = EC_anonymizer(DGhs, k_min_cost)

    for k_lowest_data in k_min_cost:
        k_lowest_data["check"] = True

    for r in customized_dataset:
        r["cost"] = 0



def add_anonymized_data(raw_dataset, k_min_cost):
    for k_min_data in k_min_cost:
        raw_dataset[k_min_data["index"]] = k_min_data
    return raw_dataset


def EC_anonymizer(DGHs, EC):
    domain_name_list = list(DGHs.values())
    generalized_data = list()

    for d in domain_name_list:

        domain_name = d.name
        dom_name_dict = dict()
        nodes = list()

        for data in EC:

            if data[domain_name] in dom_name_dict:
                selected_node = dom_name_dict[data[domain_name]]

            else:
                selected_node = dfs_in_DGH(d.child, data[domain_name])
                dom_name_dict[data[domain_name]] = selected_node

            nodes.append(selected_node)

        traversed_nodes= traverse_for_generalization(nodes)
        generalized_domains = generalize_nodes(traversed_nodes)

        
        data[domain_name] = generalized_domains[0].name
        generalized_data.append(data)

    return generalized_data

###########################################
#COST HELPERS
###########################################
def add_costMD_col(anonymized_dataset):

    for a in anonymized_dataset:
        a["cost_MD"] = 0
    return anonymized_dataset


def add_costLM_col(anonymized_dataset):
    for a in anonymized_dataset:
        a["cost_LM"] = 0
    return anonymized_dataset


def find_value_of_md(name_list_domain, raw_dataset, anonymized_dataset):
    domain_name = dict()
    l = len(raw_dataset)

    for d in name_list_domain:
        for i in range(l):
            raw_domain_name = raw_dataset[i][d.name]

            if raw_domain_name in domain_name:
                raw_dataset_depth = domain_name[raw_domain_name]
            else:
                raw_dataset_depth = dfs_in_DGH(d.child, raw_dataset[i][d.name]).depth
                domain_name[raw_domain_name] = raw_dataset_depth

            anonymized_domain_name = anonymized_dataset[i][d.name]

            if anonymized_domain_name in domain_name:
                anonymized_ds_depth = domain_name[anonymized_domain_name]
            else:
                anonymized_ds_depth = dfs_in_DGH(d.child, anonymized_dataset[i][d.name]).depth
                domain_name[anonymized_domain_name] = anonymized_ds_depth

            generalization_cost = raw_dataset_depth - anonymized_ds_depth
            anonymized_dataset[i]["cost_MD"] += generalization_cost


    return anonymized_dataset

def find_table_md(anonymized_dataset):
    c = 0
    for anonymized_data in anonymized_dataset:
        c += anonymized_data["cost_MD"]
    return c

def find_value_lm(name_list_domain, anonymized_dataset):

    for d in name_list_domain:

        domain_name = d.name
        value_lm_dict = dict()
        total_leaves = d.child.count_leaves()

        for done in anonymized_dataset:

            anon_node = dfs_in_DGH(d.child, done[domain_name])

            if anon_node.name in value_lm_dict:
                val_lm = value_lm_dict[domain_name]

            else:

                l = len(anon_node.child)
                if l == 0:
                    numerator = 0

                else:
                    numerator = anon_node.count_leaves() - 1

                denominator = total_leaves - 1
                val_lm = numerator / denominator
                value_lm_dict[domain_name] = val_lm

            done["cost_LM"] += val_lm

    return anonymized_dataset


def find_records_lm(anonymized_dataset, a):
    for done in anonymized_dataset:
        done["cost_LM"] *= (1 / a)
    return anonymized_dataset


def find_table_lm(anonymized_dataset):
    c = 0
    for done in anonymized_dataset:
        c += done["cost_LM"]
    return c