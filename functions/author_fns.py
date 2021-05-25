import numpy as np
import igraph

class Author:
    def __init__(self, network_bias:float, walk_bias:float, meet_bias:float, beta:float, gender='W'):
        '''
        Inputs:
            gender          string, 'W' or 'M'
            network_bias    a scalar float. Determines how much their estimated social notwork 
                            is biased towards people with the same gender. 0 men will be chosen 
                            whenever possible, and 1 indicates women will be hcosen whenever possible
            walk_bias       a scalar floar. Determines how much people will bias themselves towards 
                            women in their citations. 1 indicates they will cite women whenver possible,
                            0 indicates they will cite men whenever possible
            meet_bias       a scalar float. Determines how similar a strangers network has to 
                            be to meet with them
            beta            a scalar float. Determines how this author learns from others. The name beta
                            comes from a model sequence learning, where the parameter beta tuned the 
                            steepness of temporal discounting
        '''
        # type checking
        if not(isinstance(beta,float)):
            raise Exception('Beta should be a float')
        if not(isinstance(meet_bias,float)):
            raise Exception('Meeting bias should be a float')
        if not(isinstance(network_bias,float)):
            raise Exception('Network bias should be a float')
        if not(isinstance(walk_bias,float)):
            raise Exception('Walk bias should be a float')
        if not(isinstance(gender,str)):
            raise Exception('Gender should be a string, W or M')
        
        # value checking
        gender = gender.upper()
        if (gender != 'W') & (gender != 'M'):
            raise Exception('Gender should be W or M')
        if (beta < 0.) | (beta > 1.):
            raise Exception('Beta should be between 0 and 1')
        if (meet_bias < 0.) | (meet_bias > 1.):
            raise Exception('Meeting bias should be between 0 and 1')
        if (network_bias < 0.) | (network_bias > 1.):
            raise Exception('Network bias should be between 0 and 1')
        if (walk_bias < 0.) | (walk_bias > 1.):
            raise Exception('Walk bias should be between 0 and 1')    
           
        self.gender = gender
        self.learn_bias = beta
        self.meet_bias = meet_bias
        self.network_bias = network_bias
        self.walk_bias = walk_bias
        
    def init_network(self, g, node2gen, n, d=3, c=3):
        '''
        This will initialize the author's estimate of the social network. The network will be biased to have more 
        women/men based on the network bias parameter. The network will be created from a Levy flight starting from 
        a random node. This is to assure that the nextowrk stays mostly connected
        (adapted from https://github.com/dalejn/kinestheticCuriosity/blob/master/scripts/errwLevyFunction.py)
    
        Inputs:
        g               the co-author network as an igraph object
        node2gen        a dictionary mapping node indices to gender information
        n               int: the number of nodes in the subsampled graph
        d            int > 0: the diameter, or maximum step allowed in the levy flight 
                        (step of 1 will just be a random walk)
        c               float: the levy coefficient (1 < c <= 3)
        
        New parameters:
        self.network    the estimated social network of this individual
        '''
        # check that your param numbers are reasonable
        if len(g.clusters().giant().vs()) <= n:
            raise Exception('You have requested a network bigger than the largest connected component')
        if (c < 1) | (c > 3):
            raise Exception('Your levy coefficient should be between 1 and 3')
        if (d < 1):
            raise Exception('Your diameter should be greater than 0')
        
        # get decaying distribution of step sizes
        x = np.arange(1, d+1, 1)  # set upper bound for step size
        pdfLevy = np.power(1/x, c)  # set u value, where 1 <= u <= 3
        pdfLevy = pdfLevy/sum(pdfLevy)

        # intialize
        startCoord = np.random.choice(g.vs.select(_degree_gt = 0)).index 
        # must have big enough connected component to get the desired size
        found_st = False
        while not found_st:
            # check if the connected component for this node is greater than 
            c_idx = [startCoord in x for x in g.clusters()]
            if len(g.clusters()[np.where(c_idx)[0]]) >= n:
                found_st = True
            else:
                startCoord = np.random.choice(g.vs.select(_degree_gt = 0)).index
            
        nodes = {}
        gender = []
        nodes[startCoord] = 0
        gender.append(node2gen[startCoord]['gender'])
        
        # get the walk
        k = 1
        while k < n:
            # get step size from our exp distribution
            stepSize = np.random.choice(x, p=pdfLevy)
            sourceNodes = []
            sourceNodes.append(startCoord)

            # pick a node of the proper step size
            for step in range(0, stepSize):
                nextNode = g.neighbors(sourceNodes[step])
                if len(nextNode) == 1:
                    sourceNodes.extend(nextNode)
                else:
                    # get transition probabilities, weighted by gender (but only if this is the last step)
                    transitionProb = np.ones((len(nextNode),))
                    gen_idx = [node2gen[x]['gender'] for x in nextNode]
                    # check if everyone is the same gender, or if we aren't on the last one yet
                    if (len(set(gen_idx)) == 1) | (step != (stepSize - 1)):
                        transitionProb = transitionProb*(1/len(transitionProb))
                    else:
                        # low network bias parameters bias towards men
                        transitionProb[[x != 'woman' for x in gen_idx]] = 1 - self.network_bias
                        transitionProb[[x == 'woman' for x in gen_idx]] = self.network_bias
                        # normalize so it sums to 1
                        if (self.network_bias != 1.0) & (self.network_bias != 0.0):
                            transitionProb = transitionProb/sum(transitionProb)
                        else:
                            transitionProb = [x/sum(transitionProb) for x in transitionProb]
                    sourceNodes.append(np.random.choice(nextNode, p=transitionProb))
            
            # only add to our list if we havent walked on this node before
            if sourceNodes[-1] not in nodes.keys():
                nodes[sourceNodes[-1]] = k
                gender.append(node2gen[sourceNodes[-1]]['gender'])
                k += 1
            startCoord = sourceNodes[-1]
            
        # now, make a new network from this walk
        sg = igraph.Graph()
        sg.add_vertices(nodes.values())
        # oid is the original id in the full coauthor graph
        sg.vs['oid'] = [key for key in nodes.keys()] # google said that keys() and values() calls would always be corresponding
        sg.vs['gender'] = gender
        # get unique edges
        es = list(set([edge.tuple for edge in g.es.select(_within=list(nodes.keys()))]))
        # map from original to new IDs
        for i,e in enumerate(es):
            v1 = nodes[e[0]]
            v2 = nodes[e[1]]
            es[i] = (v1,v2)
        sg.add_edges(es)
        self.network = sg


        
    def get_cites(self, n, start=None):
        '''
        This will generate a simulated citation list from a random walk on the graph, 
        biased towards nodes of a given gender based on the
        Inputs:
        n               the number of names in the citation list
        start           if desired, the node ID to start at. Otherwise will be chosen randomly
        
        Returns:
        bib             A dictionary where keys are IDs and lavues are the names and genders of the 
                        simulated citation list
        '''
        
        # type and value checking for n
        if not(isinstance(n,int)):
            raise Exception('N should be an integer')
        if n < 1:
            raise Exception('N should be an integer greter than 0')
        if len(self.network.vs.select(_degree_gt = 1)) == 0:
            raise Exception('Your network is disconnected - something went wrong with your network intialization')
        
        # initialize
        bib = {}
        
        # useful constants
        nNode = len(self.network.vs())
        
        if start == None:
            # make sure the starting node is connected to something
            start = np.random.choice(self.network.vs.select(_degree_gt = 1)).index
        bib[0] = {'id': start,
                  'gender':self.network.vs['gender'][start],
                  'oid':self.network.vs['oid'][start]}
        
        if n > 1:
            for i in range(1, n-1):
                nodes = self.network.neighbors(bib[i-1]['id'])
                # get transition probabilities, weighted by gender 
                transitionProb = np.ones((len(nodes),))
                gen_idx = self.network.vs.select(nodes)['gender']
                # check if everyone is the same gender
                if (len(set(gen_idx)) == 1):
                    next_node = np.random.choice(nodes)
                else:
                    # low walk bias parameters bias towards men
                    transitionProb[[x != 'woman' for x in gen_idx]] = 1 - self.walk_bias
                    transitionProb[[x == 'woman' for x in gen_idx]] = self.walk_bias
                    # normalize so it sums to 1
                    if (self.network_bias != 1.0) & (self.network_bias != 0.0):
                        transitionProb = transitionProb/sum(transitionProb)
                    else:
                        transitionProb = [x/sum(transitionProb) for x in transitionProb]
                    next_node = np.random.choice(nodes, p=transitionProb)

                bib[i] = {'id': next_node,
                      'gender':self.network.vs['gender'][next_node],
                      'oid':self.network.vs['oid'][next_node]}  

        return bib
    
    def update_network(self, walk, g, thr=.1, debug=False):
        '''
        This function will update and Author object's network based on a specific walk. 
        The fidelity of the update will depend on the author's learn_bias parameter
        
        Inputs:
        walk     a dictionary of nodes traversed in a walk (like what is output by self.get_cites())
                 the dictionary should contain the ID of the node in the original co-author network
        g        an igraph graph object that generated the walk used for the first input (the .network 
                 property of another author)
        thr      the threshold to cut edges out of the A_hat matrix
        debug    if True, it will return the matrices from internal steps
        '''
        
        # first, get true transition structure from the original network
        nodes = [val['id'] for val in walk.values()]
        unodes = list(set(nodes))
        sg = g.subgraph(g.vs(unodes))
        A = sg.get_adjacency()
        A = np.array(A.data)
        A = np.divide(A,A.sum(0))
        
        # incorporate betas to get A_hat
        A_hat = np.matmul((1 - np.exp(-self.learn_bias))*A,np.linalg.pinv(np.eye(len(unodes)) - np.exp(-self.learn_bias)*A))

        # threshold
        A_hatb = A_hat.copy()
        A_hatb[A_hatb < thr] = 0
        A_hatb[A_hatb > 0] = 1
        # get nodes with no edges
        idx = np.where(A_hatb.sum(0) == 0)[0]
        sg.delete_vertices(idx.tolist())
        for e in sg.es():
            v1 = e.tuple[0]
            v2 = e.tuple[1]
            if A_hatb[v1,v2] == 0:
                sg.delete_edges(e)
        
        # combine
        # first add new nodes
        for v in sg.vs():
            if v['oid'] not in self.network.vs['oid']:
                self.network.add_vertices(1, attributes={'oid':v['oid'],
                                                          'gender':v['gender']})
        
        # then add edges 
        for e in sg.es():
            oid_tuple = (sg.vs(e.tuple[0])['oid'][0], sg.vs(e.tuple[1])['oid'][0])
            v1 = self.network.vs.select(oid_eq=oid_tuple[0])
            v2 = self.network.vs.select(oid_eq=oid_tuple[1])
            if (len(v1) > 1) | (len(v2) > 1):
                raise Exception('Something went wrong - you have multiple nodes with the same oid')
            new_e = (v1[0].index,v2[0].index)
            self.network.add_edges([new_e])
        
        if debug:
            return A_hat, A, A_hatb, sg
        else:
            return self

    
def compare_nets(a1,a2):
    '''
    This function will take two author objects, and determine if thier network's are similar enough to meet
    Inputs:
    a1    an Author object, with a meet bias and a network
    a2    an Author object with a meet bias and netork
    
    Returns:
    meet12     a boolean, True is a1 will meet with a2, False otherwise
    meet21     a boolean, True is a2 will meet with a1, False otherwise
    '''
    
    # type checking
    if not ((isinstance(a1,Author)) and isinstance(a2,Author)):
        raise Exception('You must input two Author objects into this function')
        
    # get node identities
    n1 = a1.network.vs['oid']
    n2 = a2.network.vs['oid']
    
    # compare node identities
    meet12 = sum([x in n2 for x in n1])/len(n1) > a1.meet_bias
    meet21 = sum([x in n1 for x in n2])/len(n2) > a2.meet_bias
    
    return meet12, meet21