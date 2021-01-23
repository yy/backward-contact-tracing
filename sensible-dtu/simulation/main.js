const ArgumentParser = require('argparse').ArgumentParser;
const log = require('../src/log');
const io = require('../src/io');
const SEIRSimple = require('../src/seir-simple');


// TODO Process results: curves (s, e, i, r), outbreak size dist.
// TODO Process results: p(k) of infected until some time, p(k) of parents until some time.
// TODO Process results: phi(t, t + dt).
// TODO Process results: p(k) of traced nodes with p_r, p_t, p_s.

// Read arguments.
const parser = new ArgumentParser({
    version: '0.0.1',
    addHelp: true,
    description: 'Simple SEIR simulation (node-node interactions).'
})
parser.addArgument(['--input'], {
    dest: 'input',
    required: true,
    help: 'Path to the CSV file containing the temporal links.'
})
parser.addArgument(['--output'], {
    dest: 'output',
    required: true,
    help: 'Path to the output dir containing the simulation results. Each simulation has two output files: <name>_meta.csv containing the meta data for each run and <name>_logs.csv containing the simulation event log.'
})
parser.addArgument(['--infection-rate'], {
    dest: 'infectionRate',
    type: 'float',
    defaultValue: 0.2,
    help: 'Infection rate in 1/day units [0.2].'
})
parser.addArgument(['--incubation-period'], {
    dest: 'incubationPeriod',
    type: 'float',
    defaultValue: 1,
    help: 'Incubation period in days [1].'
})
parser.addArgument(['--infectious-period'], {
    dest: 'infectiousPeriod',
    type: 'int',
    defaultValue: 14,
    help: 'Incubation period in days [14].'
})
parser.addArgument(['--index-cases'], {
    dest: 'indexCases',
    type: 'float',
    defaultValue: 0.01,
    help: 'Fraction of index cases [0.01].'
})
parser.addArgument(['--start-time'], {
    dest: 'startTime',
    nargs: '+',
    type: 'int',
    defaultValue: null,
    help: 'Start time index [null].'
})
parser.addArgument(['--laps'], {
    dest: 'laps',
    type: 'int',
    defaultValue: 10,
    help: 'Number of realizations [10].'
})
const args = parser.parseArgs();


(async () => {
    // Read data for the config.
    const simulation = await SEIRSimple().load(args.input)

    // Set simulation parameters.
    simulation.infectionRate(args.infectionRate)
        .incubationPeriod(args.incubationPeriod)
        .infectiousPeriod(args.infectiousPeriod)
        .indexCases(args.indexCases)

    // Run laps.
    for (let i = 0; i < args.laps; i++) {
        log.i(`Lap #${i}`)

        // Run simulation and take meta.
        const meta = Object.assign({
            infectionRate: args.infectionRate,
            incubationPeriod: args.incubationPeriod,
            infectiousPeriod: args.infectiousPeriod,
            indexCases: args.indexCases
        }, simulation.run(args.output, i, args.startTime))
        await io.writeCSV([meta], `${args.output}_meta.csv`, i > 0)
    }
})()
