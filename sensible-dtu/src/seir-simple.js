const { nest } = require('d3-collection');
const ran = require('ranjs');
const log = require('./log');
const io = require('./io');
const uuid = require('uuid').v1;

// TODO Run simulations with different parameters (T, R0), measure outbreak size.

/**
 * Factory for the epidemic config.
 *
 * @function SEIRSimple
 */
module.exports = () => {
    // Constants.
    const STATES = {
        // Values are chosen so that we have:
        // S * I = 2
        // E % 2 = I % 2 = 0
        susceptible: 1,
        exposed: 4,
        infectious: 2,
        removed: 3
    }
    const DT = 1. / 288

    let _ = {
        graph: {
            nodes: [],
            binnedLinks: []
        },
        parameters: {
            index: 10,
            beta: 0.01,
            gamma: 0.1,
            a: 0.1
        },
        sizes: {
            time: 0,
            links: 0,
            nodes: 0
        },
        counters: {
            timestamp: 0,
            elapsed: 0
        },
        config: {
            id: null,
            seed: 0,
            t: 0
        },
        events: []
    }

    function extractLinks(data) {
        log.i('Extracting links.')

        // Print some statistics.
        _.sizes.links = data.length
        _.sizes.time = Math.max(...new Set(data.map(d => d.ts))) + 1
        log.i(`  number of links: ${_.sizes.links}.`)
        log.i(`  number of bins:  ${_.sizes.time}.`)

        // Build binned links.
        _.graph.binnedLinks = Array.from({length: _.sizes.time})
        nest()
            .key(d => d.ts).sortKeys((a, b) => +a - +b)
            .rollup(values => values.map(d => ({
                n1: d.n1,
                n2: d.n2
            })))
            .entries(data)
            .map(d => _.graph.binnedLinks[+d.key] = d.value)
    }

    function extractNodes(data) {
        log.i('Extracting nodes.')

        // Collect IDs from both sides.
        const nodes1 = new Set(data.map(d => d.n1))
        const nodes2 = new Set(data.map(d => d.n2))

        // Initialize nodes.
        _.sizes.nodes = Math.max(...new Set([...nodes1, ...nodes2])) + 1
        _.graph.nodes = Array.from({length: _.sizes.nodes})
            .map((d, id) => ({id}))

        // Print some statistics.
        log.i(`  number of nodes: ${_.sizes.nodes}.`)
    }

    function getElapsedTime() {
        let days = Math.floor(_.counters.elapsed / (12 * 24))
        return `Day ${days}`
    }

    function addEvent(type, node, source = null) {
        _.events.push({
            id: _.config.id,
            elapsed: _.counters.elapsed,
            type,
            node,
            source
        })
    }

    function iterate() {
        // Update timestamp.
        _.counters.timestamp = (_.counters.timestamp + 1) % _.sizes.time
        _.counters.elapsed++

        // Get current S-I links.
        // First we get links in current timestamp, then we filter on the product of the states of the end nodes.
        // As node states are encoded as integers, the product should determine the link status.
        // Update S -> E.
        _.graph.binnedLinks[_.counters.timestamp]
            .filter(d => _.graph.nodes[d.n1].state * _.graph.nodes[d.n2].state === 2)
            .filter(() => ran.core.float() < _.parameters.beta)
            .forEach(d => {
                // Pick source and target.
                let source = _.graph.nodes[d.n1].state === STATES.infectious ? d.n1 : d.n2
                let target = source === d.n1 ? d.n2 : d.n1

                // Update target's next state.
                _.graph.nodes[target].next = STATES.exposed

                // Log infection event.
                addEvent('e', target, source)
            })

        // Update E -> I.
        _.graph.nodes.filter(d => d.state === STATES.exposed)
            // Select gamma fraction of the infectious who will be removed.
            .filter(() => ran.core.float() < _.parameters.a)
            .forEach(d => {
                d.next = STATES.infectious

                // Log removed event.
                addEvent('i', d.id)
            })

        // Update I -> R.
        _.graph.nodes.filter(d => d.state === STATES.infectious)
            // Select gamma fraction of the infectious who will be removed.
            .filter(() => ran.core.float() < _.parameters.gamma)
            .forEach(d => {
                d.next = STATES.removed

                // Log removed event.
                addEvent('r', d.id)
            })

        // Update states and remove next.
        _.graph.nodes.filter(d => d.next)
            .forEach(d => {
                d.state = d.next
                delete d.next
            })
    }

    function init(t, seed) {
        // Set ID and seed.
        _.config.id = uuid()
        _.config.seed = seed || Date.now()

        // Set start time.
        _.config.t = t === null ? ran.core.int(_.sizes.time) : t.length === 2 ? ran.core.int(t[0], t[1]) : t[0]

        // Set seed.
        ran.core.seed(_.config.seed)

        // Initialize states.
        _.graph.nodes = _.graph.nodes.map(d => Object.assign(d, {
            state: STATES.susceptible
        }))

        // Infect some fraction of the nodes.
        let numInfected = _.parameters.index
        while (numInfected > 0) {
            // Choose random node from the among susceptible.
            let node = ran.core.choice(_.graph.nodes.filter(d => d.state === STATES.susceptible))
            node.state = STATES.exposed
            numInfected--
        }

        // Add first events.
        _.events = _.graph.nodes.filter(d => d.state === STATES.exposed).map(d => ({
            id: _.config.id,
            elapsed: 0,
            type: 'e',
            node: d.id,
            source: null
        }))

        // Reset counters.
        _.counters.elapsed = 0
        _.counters.timestamp = _.config.t
    }

    // Public methods.
    let api = {}

    /**
     * Loads the temporal network data as a CSV file.
     *
     * @method load
     * @methodOf SEIRSimple
     * @param {string} path Path to the CSV file containing the temporal links.
     * @param {number} [rssiMin = -100] Lower boundary of RSSI values to accept a bluetooth signal.
     * @returns {Promise<SEIRSimple>} Promise containing the SEIRSimple API.
     */
    api.load = async path => {
        log.i('Loading network data.')

        // Load and map data.
        let data = await io.readCSV(path).catch(log.e)
        data = data.map(d => ({
            ts: +d['timestamp'],
            n1: +d['n1'],
            n2: +d['n2']
        }))

        // Extract links.
        extractLinks(data)

        // Extract IDs from both sides.
        extractNodes(data)

        // Calculate some metrics.
        log.i(`Temporal average degree: ${2 * data.length / (_.sizes.nodes * _.sizes.nodes)}`)

        return api
    }

    /**
     * Sets the config infection probability corresponding the specified physical rate of infection in 1/day.
     *
     * @method infectionRate
     * @methodOf SEIRSimple
     * @param {number} beta Physical rate of infection in 1/day.
     * @returns {SEIRSimple} Reference to the SEIRSimple API.
     */
    api.infectionRate = (beta = 0.1) => {
        // Calculate temporal average degree.
        const k = 2 * _.sizes.links / (_.sizes.nodes * _.sizes.time)

        // Set config beta = beta_phys * dt / k
        _.parameters.beta = beta * DT / k
        log.i(`Infection rate is set to ${_.parameters.beta.toPrecision(2)} (beta = ${beta} 1/day)`)
        return api
    }

    /**
     * Sets the config inverse incubation period corresponding the specified physical incubation period in days.
     *
     * @method incubationPeriod
     * @methodOf SEIRSimple
     * @param {number} T Expected incubation period in days.
     * @returns {SEIRSimple} Reference to the SEIRSimple API.
     */
    api.incubationPeriod = (T = 14) => {
        // Set config gamma = dt / T_phys
        _.parameters.a = DT / T
        log.i(`Incubation rate is set to ${_.parameters.a.toPrecision(2)} (T = ${T} day)`)
        return api
    }

    /**
     * Sets the config recovery probability corresponding the specified physical infectious period in days.
     *
     * @method infectiousPeriod
     * @methodOf SEIRSimple
     * @param {number} T Expected infectious period in days.
     * @returns {SEIRSimple} Reference to the SEIRSimple API.
     */
    api.infectiousPeriod = (T = 3) => {
        // Set config gamma = dt / T_phys
        _.parameters.gamma = DT / T
        log.i(`Infectious rate is set to ${_.parameters.gamma.toPrecision(2)} (T = ${T} day)`)
        return api
    }

    /**
     * Sets the fraction of index cases.
     *
     * @method indexCases
     * @methodOf SEIRSimple
     * @param {number} i0 Fraction of index cases.
     * @returns {SEIRSimple} Reference to the SEIRSimple API.
     */
    api.indexCases = (i0 = 0.01) => {
        _.parameters.index = Math.round(i0 * _.sizes.nodes)
        log.i(`Number of index cases is set to ${_.parameters.index} (i = ${Math.round(100 * i0)}%)`)
        return api
    }

    /**
     * Runs the config until there are no infectious nodes.
     *
     * @method run
     * @methodOf SEIRSimple
     */
    api.run = (path, index, t0, seed) => {
        // Initialize.
        init(t0, seed)

        // Run simulation.
        while (_.graph.nodes.filter(d => d.state % 2 === 0).length > 0) {
            iterate()
        }

        // Save events.
        io.writeCSV(_.events, `${path}_logs.csv`, index > 0)

        // Return config settings.
        return _.config
    }

    return api
}
