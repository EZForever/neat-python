"""
hyperneat implementation for neat-python.
"""
import copy
import random
import collections
import itertools
import math

import neat
from neat import genome, config, population, genes, attributes


"""
Planning
=========
How a HyperNEAT works:
----------------------
1. each node has a position (e.g. (x, y))
2. each connection can be represented as the coordinate between the start node
    and the out node (e.g. x1, y1, x2, y2))
3. the nodes are seperated into layers. Each layer has a its own coordinate
    system. Typical layers are: input layer, hidden layer, output layer
    (todo: check if it is possible to have any number of hidden layers).
4. the NEAT network which is seperated into these layers is called 'substrate'
    or 'ANN' (artificial neural network).
5. There is a second NEAT (not HyperNEAT!) called the 'CPPN'
    (compositional pattern production network).
    This NEAT has n*2 inputs (the connection), where
    n=number_of_dimensions_of_a_layer and 1 output
    the weight of the connection).
    In more advanced versions, the CPPN has 2 outputs, the second one being the
    bias of a node. However, as the CPPN has 2*n inputs and only n inputs are
    required to describe the location of a node, a convention is needed to
    determine the input representation of a node. A suggestion is setting the
    inputs for the end node to 0. @drallensmith suggested setting both positions
    to the position of the node.
6. when evaluating the substrate, start a neat evolution of the CPPN with
    the substrate. The fitness of the substrate becomes the fitness of the CPPN.
    The champion CPPN is then used to evaluate the fitness of the substrate and
    the resulting fitness becomes the fitness of the substrate.
    During all evaluations of the substrates, the connection weight and node
    bias is determined by activating the CPPN. connections with a weight < 0.0
    will be disabled.

----------------
Problems
----------------
1. Placement of nodes:
    Nodes need to be placed in a specific position.
    Solutions:
        Hidden nodes:
            1.1 use es-NEAT (follow information density
            1.2 random evolution like response
            1.3 make a switch in the config between these options
        Input nodes:
            1.4 use a new node class
            1.5 write a function to auto-place input-nodes in the input layer
                1.5.1 (v1, v2, v3) --> (v1[0], v2[1], v3[2])
                1.5.2 (
                    (v1, v2),
                    (v3, v4),
                    )   --> (v1[0, 0], v2[0, 1], v3[1, 0], v4[1, 1])
        Output nodes:
            1.6 maybe use a attribute to define the placement?

2. Compatibility with existent NN implementations in neat-python:
    We should probably try to make the current network implementations
    (feedforward, iznn, ...) compatible with hyperneat.
    2.1 rewrite these networks to support hyperneat
    2.2 write the hyperneat implementation compatible with the other networks
        2.2.1 rewrite the genomes to use a @property for weight and bias.
    2.3 write new implementations of the networks.
-----------------
additional goals
-----------------
1. Make it scallable (configuartion wise):
    1.1 each layer (and the position) should be able to have any number of
        dimensions (but each needs to have the same number)
    1.2 if possible, make the number of hidden layers configurable
    1.3 test if it is possible to make the CPPN an hyperneat itself.
2. Make it configurable:
    2.1 the points in 1. should be configurable
    2.2 the config of the CPPN should be fully configurable (except dimensions)
3. implement some HyperNEAT extension algorithms:
    3.1 es-hyperneat (automatic hidden node placement)
    3.2 hyperneat-LEO (disable connections with a CPPN expression value <= 0)
-----------------
TODO
-----------------
1. position determination
    1.1 config
    1.2 es-hyperneat
    1.3 mutation
2. quadtree algorithm
    2.1 needs to support any number of dimensions (k-d-tree?)
3. autoplacement of nodes (see problems 1.5)
4. tests
5. compatibility with iznn and other neural networks
6. more/better configuration options
7. optimize CPPN
    7.1 make LEO and bias disable-able
    7.1 reduce number of outputs to the required minimum
    (if we should not determine the bias, we do not need this output)
8. check names for config options and other names
"""


# constants for easier access to the input and output layers
INPUT_LAYER = 0  # 0 because the index 0 points to the first element
OUTPUT_LAYER = -1  # -1 because the index -1 points to the last element

class DimensionError(ValueError):
    """
    Exception raised when some dimensions mismatch.
    An example is the distance between the points (1, 4) and (3, 8, 4).
    """
    pass


class Coordinate(object):
    """This represents a n-dimensional coordinate"""
    __slots__ = ("values", "n")
    def __init__(self, values):
        self.values = list(values)
        self.n = len(values)

    def expand(self, value, index=0):
        """expands the dimensionality of the coordinate"""
        self.values.insert(index, value)
        self.n += 1
    
    def zero_copy(self):
        """returns a copy of this coordinate at (0, 0, ...)"""
        return self.__class__([0] * self.n)

    def __len__(self):
        """returns the number of dimensions this coordinate has"""
        return self.n

    def __reduce__(self):
        """used by pickle to pickle this class"""
        # we do not need this, but this may be a bit more space efficient.
        return (self.__class__, (self.values, ))

    def __iter__(self):
        """iterate trough the coordinates"""
        for v in self.values:
            yield v

    def distance_to(self, other):
        """
        returns the distance between this coordinate and the other coordinate
        """
        if self.n != other.n:
            raise DimensionError(
                "can not calculate distance between points of different \
                dimensionality!"
            )
        deltas = [sv - ov for sv, ov in zip(self.values, other.values)]
        qs = sum([d ** 2 for d in deltas])
        return math.sqrt(qs)




class _QuadPoint(object):
    """
    A node in the quadtree.
    TODO: rename this. i am not sure if the modified algorithm is still a
    quadtree. IIRC the algorithm and points are only called a quadthree when
    they are two-dimensional.
    """
    __slots__ = ["position", "width", "level", "cs"]
    def __init__(self, position, width, level):
        assert isinstance(position, Coordinate)
        self.position = position
        self.width = width
        self.level = level
        self.cs = [None] * (2 ** position.n)

class CPPNGenome(genome.DefaultGenome):
    """
    A neat.genome.DefaultGenome subclass for the CPPN.
    We need this class for LEO seeding.

    LEO CPPN network:
    - each coordinate input nodes (x1 and x2, y1 and y2 ...) are each connected
    to a hidden node with a gaussian activation function. The weight of
    (x1, y1...) is positive, while the weight of (x2, y2 ...) is negative.
    These hidden nodes are connected to the LEO which has a bias of -3.
    """
    pass


class HyperGenomeConfig(genome.DefaultGenomeConfig):
    """A neat.genome.DefaultGenome subclass for HyperNEAT"""
    _params_sig = genome.DefaultGenomeConfig._params_sig + [
        config.ConfigParameter("cppn_config", str, None),
        config.ConfigParameter("cppn_generations", int, 30),
        # TODO: find a better name for the following ConfigParameters
        config.ConfigParameter('allow_connections_to_lower_layers', bool, False),
        config.ConfigParameter("substrate_dimensions", int, 2),
        config.ConfigParameter("leo_threshold", float, 0.45),
        ]


class HyperGenome(genome.DefaultGenome):
    """A neat.genome.DefaultGenome subclass for HyperNEAT"""

    # arguments passed to the cppn config.Config()
    cppn_genome = CPPNGenome  # neat.DefaultGenome
    cppn_reproduction = neat.DefaultReproduction
    cppn_species_set = neat.DefaultSpeciesSet
    cppn_stagnation = neat.DefaultGenome

    def __init__(self, key):
        genome.DefaultGenome.__init__(self, key)
        self.cppn_pop = None
        self.cppn = None
        self._did_evolve_cppn = False
        self._fitness_function = None

    @classmethod
    def parse_config(cls, param_dict):
        param_dict['node_gene_type'] = HyperNodeGene
        param_dict['connection_gene_type'] = HyperConnectionGene
        dimensions = int(param_dict.get("substrate_dimensions", 2))
        param_dict["num_inputs"] = (dimensions * 2) + 1
        param_dict["num_outputs"] = 3
        hgc = HyperGenomeConfig(param_dict)
        if hgc.cppn_config is None:
            raise RuntimeError("No CPPN config specified!")
        return hgc

    @property
    def fitness_function(self):
        """the fitness function used for the currenty evaluation"""
        return self._fitness_function

    @fitness_function.setter
    def fitness_function(self, value):
        """the fitness function used for the currenty evaluation"""
        self._fitness_function = value
        if not self._did_evolve_cppn:
            self._evolve_cppn()

    def _create_cppn_pop(self, config):
        """creates a new CPPN population"""
        config = config.Config(
            self.cppn_genome,
            self.cppn_reproduction,
            self.cppn_species_set,
            self.cppn_stagnation,
            self.config.cppn_config,
            )
        pop = population.Population(config)
        return pop

    def _evolve_cppn(self):
        """evolves the cppn"""
        if self._fitness_function is None:
            raise RuntimeError("Fitness function is not set!")

        def _tff(genomes, config):
            """
            A replacement for the fitness function.
            We need to determine the fitness of the CPPN with the fitness
            function of the substrate.
            """
            copies = []
            for i, (genome_id, genome_) in enumerate(genomes):
                c = copy.copy(self)  # a shallow copy should be enough
                c.cppn = genome_
                copies.append(genome_id, c)
            self._fitness_function(copies, config)

        # determine the winner
        winner = self._cppn_pop.run(_tff, self.config.genome_config.cppn_generations)
        # we set this genomes cppn to the winner
        self.cppn = winner
        self._did_evolve_cppn = True

    def activate_cppn(self, p1, p2):
        """returns the (weight, expression, bias) for the connection or gene"""
        # convention: when querrying for bias, p2 == p1
        # TODO: check if it makes sense to supply the layer index to the CPPN
        l = p1.distance_to(p2)
        # create a CPPN network
        # TODO: make network type configurable
        net = neat.nn.FeedForwardNetwork.create(self.cppn, self._cppn_config)
        # set inputs
        # we supply the following inputs:
        # - positions of both nodes
        # - distance between them
        # TODO: some papers seem to use a 'bias' value. Which bias is meant?
        # EDIT: seems like this bias is an alternative to the internal bias used
        # by neat-python nodes
        inp = list(p1) + list(p2) + [l]
        weight, expression, bias = net.activate(inp)
        return weight, expression, bias

    def weight_between_points(self, p1, p2):
        """returns the weight for the connection between the specified points"""
        return self.activate_cppn(p1, p2)[0]

    def weight_for_connection(self, conn):
        """returns the weight for the specified connection"""
        i, o = conn._get_nodes()
        p1 = i.position
        p2 = o.position
        return self.weight_between_points(p1, p2)
    
    def expression_value_for_connection(self, conn):
        """returns the expression value for the specified connection"""
        # TODO: use value from CPPN
        # it appears that leo requires an premodified CPPN.
        #
        # we use the formula from
        # http://eplex.cs.ucf.edu/papers/verbancsics_gecco11.pdf
        # t = c + c * |ni -nj|
        # where t = the local threshold
        # and c = a global defined value (defined in the config)
        # and |ni - nj| the distance between the nodes
        i, o = conn._get_nodes()
        p1 = i.position
        p2 = o.position
        d = p1.get_distance_to(p2)
        c = self.config.genome_config.leo_threshold
        t = c + c * d
        return t

    def bias_for_node(self, p):
        """returns the bias for the node at p"""
        return self.activate_cppn(p, p)[2]

    def configure_new(self, config):
        """configures a new HyperGenome from scratch."""
        genome.DefaultGenome.configure_new(self, config)
        self.config = config
        self.cppn_pop = self._create_cppn_pop()

    def configure_crossover(self, genome1, genome2, config):
        """ Configure a new genome by crossover from two parent genomes. """
        genome.DefaultGenome.configure_crossover(self, genome1, genome2, config)
        self.config = config
        if genome1.fitness > genome2.fitness:
            parent1, parent2 = genome1, genome2
        else:
            parent1, parent2 = genome2, genome1
        self.cppn_pop = parent1.cppn_pop

        self._evolve_cppn()

    def add_connection(self, config, input_key, output_key, weight, enabled):
        """adds a new connection"""
        sn = self.nodes[input_key]
        en = self.nodes[output_key]
        if not config.genome_config.allow_connections_to_lower_layers:
            assert (en.layer >= sn.layer) or (en.layer == OUTPUT_LAYER)
        connection = genome.DefaultGenome.add_connection(
            self,
            config,
            input_key,
            output_key,
            weight,
            enabled,
            )
        connection.genome = self

    def create_connection(self, config, input_id, output_id):
        """creates a new connection"""
        sn = self.nodes[input_id]
        en = self.nodes[output_id]
        if not config.genome_config.allow_connections_to_lower_layers:
            assert (en.layer >= sn.layer) or (en.layer == OUTPUT_LAYER)
        connection = genome.DefaultGenome.create_connection(
            config,
            input_id,
            output_id,
            )
        connection.genome = self
        return connection

    def create_node(self, config, node_id):
        """creates a new node"""
        node = genome.DefaultGenome.create_node(config, node_id)
        node.genome = self
        return node

    def mutate_add_connection(self, config):
        """
        Attempt to add a new connection, the only restrictions being that the
        output node cannot be one of the network input pins and the output node
        is not on a lower layer than the input node.
        """
        while True:
            cg = genome.DefaultGenome.mutate_add_connection(self, config)
            input_id, output_id = cg.key
            sn = self.nodes[input_id]
            en = self.nodes[output_id]
            if not config.genome_config.allow_connections_to_lower_layers:
                if (en.layer < sn.layer) and not (en.layer == OUTPUT_LAYER):
                    continue
            return cg

    def get_nodes_in_layer(self, layer):
        """returns a list of nodes in the specified layer"""
        return [node for node in self.nodes if node.layer == layer]

    def _division_and_initialization(
        self,
        position,
        outgoing,
        max_depth,
        division_threshold,
        initial_depth,
        ):
        """
        Returns a Quadtree, in which each quadnode at a position stores CPPN
        activation level for its position.
        Arguments:
            position: coordinates of source or target.
            outgoing: wether position points to source (True) or target (False).
            max_depth, division_threshold, initial_depth: ?
        """
        root = _QuadPoint(position.zero_copy(), 1, 1)
        q = collections.deque()
        q.append(root)
        while len(q) > 0:
            p = q.popleft()  # TODO: check if q access should be FIFO or LIFO
            # Divide into sub-regions and assign children to parent
            hw = p.width / 2.0
            ops = list(itertools.permutations(["+", "-"]))
            ops += ["-"] * p.position.n + ["+"] * p.position.n
            for i in range(2 ** p.position.n):
                nv = [None] * p.position.n
                for vi, op in enumerate(ops):
                    tnv = p.position.values[vi]
                    if op == "+":
                        tnv += hw
                    else:
                        tnv -= hw
                    nv[vi] = tnv
                np = Coordinate(nv)
                p.cs[i] = _QuadPoint(np, hw, p.level + 1)
            for c in p.cs:
                if outgoing:
                    c.weight = self.weight_between_points(position, c.position)
                else:
                    c.weight = self.weight_between_points(c.position, position)
            # Divide until initial resolution or if variance is still high
            # TODO: write variance(p)
            tr = (p.level < max_depth and variance(p) > division_threshold)
            if (p.level < initial_depth) or tr:
                for child in p.cs:
                    q.append(child)
        return root
    
    def pruning_and_extraction(self, position, connections, p, outgoing):
        """
        Adds the connections that are in bands of the two-dimensional
        cross-section of the hypercube containing the source or target node to
        the connections list.
        Arguments:
            position: coordinates of source or target
            connections: list of connections (?)
            outgoing: wether position points to source (True) or target (False).
            p: initialized quadtree 
        TODO: check docstring. This docstring is from some pseudo code from a
        paper by Sebastion Risi and Kenneth O. Stanley. Since this code is
        modified to work with any number of dimensions, this docstring may be
        wrong.
        """
        # traverse quadtree depth-first
        for c in p.cs:
            if variance(c) >= variance_threshold:
                self.pruning_and_extraction(position, connections, c, outgoing)
            else:
                # Determine if point is in a band by checking neighbor CPPN values
                d = {}
                if outgoing:
                    pass


class HyperNodeGene(genes.DefaultNodeGene):
    """A subclass of genes.DefaultNodeGene for HyperNEAT"""
    _gene_attributes = genes.DefaultNodeGene._gene_attributes + [
        attributes.IntAttribute("layer"),
    ]

    def __init__(self, *args, **kwargs):
        genes.DefaultNodeGene.__init__(self, *args, **kwargs)
        self.genome = None
        self.position = None
        self._layer = None

    @property
    def bias(self):
        """the bias of the node"""
        if self.genome is None:
            raise RuntimeError("bias querried but genome not set!")
        return self.genome.bias_for_node(self.position)

    @bias.setter
    def bias(self, value):
        """the bias of the node"""
        # we get the bias from the CPPN, so we do not set it here
        pass

    @property
    def layer(self):
        """
        The layer this nodes is part of.
        Connections between nodes on the same layer behave normally, while
        connections between nodes on different layers have their attributes
        defined by the CPPN.
        """
        if self.key in self.genome.config.genome_config.input_keys:
            # node is input node
            return INPUT_LAYER
        if self.key in self.genome.config.genome_config.output_keys:
            # node is output node
            return OUTPUT_LAYER
        # node is a hidden node
        return min(1, self._layer)

    @layer.setter
    def layer(self, value):
        """
        The layer this nodes is part of.
        Connections between nodes on the same layer behave normally, while
        connections between nodes on different layers have their attributes
        defined by the CPPN.
        """
        self._layer = value

    def copy(self):
        """creates a copy of this object"""
        c = genes.DefaultNodeGene.copy(self)
        c.genome = self.genome
        c.position = self.position
        return c

    def crossover(self, gene2):
        """ Creates a new gene randomly inheriting attributes from its parents."""
        n = genes.DefaultNodeGene.crossover(self, gene2)
        n.genome = self.genome
        if random.random() > 0.5:
            n.position = self.position
        else:
            n.position = gene2.position
        return n


class HyperConnectionGene(genes.DefaultConnectionGene):
    """A subclass of genes.DefaultConnectionGene for HyperNEAT"""
    def __init__(self, *args, **kwargs):
        genes.DefaultConnectionGene.__init__(self, *args, **kwargs)
        self.genome = None
        self._weight = None
        self._enabled = True

    def _get_nodes(self):
        """returns the connected nodes"""
        i, o = self.key
        ing, ong = self.genome.nodes[i], self.genome.nodes[o]
        return ing, ong

    @property
    def weight(self):
        """the weight of the connection"""
        if self.genome is None:
            raise RuntimeError("weight querried but genome not set!")
        ing, ong = self._get_nodes()
        if ing.layer != ong.layer:
            return self.genome.weight_for_connection(self)
        else:
            return self._weight

    @weight.setter
    def weight(self, value):
        """the weigh of the connection"""
        self._weight = value

    @property
    def enabled(self):
        """wether the connection is enabled or not"""
        # TODO: check if it would be useful to set a min value in the config
        ing, outg = self._get_nodes()
        if ing.layer != outg.layer:
            e = self.genome.expression_value_for_connection(self)
            return (e > 0.0)
        else:
            return self._enabled

    @enabled.setter
    def enabled(self, value):
        """wether the connection is enabled or not"""
        self._enabled = value
