const { nest } = require('d3-collection');
const ArgumentParser = require('argparse').ArgumentParser;
const log = require('../src/log');
const io = require('../src/io');


function countTimestamps(data, column) {
    return nest()
        .key(d => d[column])
        .rollup(values => [...new Set(values.map(d => d['timestamp']))])
        .entries(data)
        .reduce((acc, d) => {
            acc[d.key] = d.value
            return acc
        }, {})
}

function filterByQuality(data, quality) {
    log.i('filtering by quality')

    // Collect total number of timestamps.
    const numTimestamps = new Set(data.map(d => d['timestamp'])).size

    // Collect distinct timestamps for each node.
    const timestampsA = countTimestamps(data, 'user_a')
    const timestampsB = countTimestamps(data, 'user_b')

    // Merge timestamp set for each node.
    const nodeQualities = [...new Set(Object.keys(timestampsA).concat(Object.keys(timestampsB)))]
        .map(d => ({
            id: d,
            quality: new Set((timestampsA[d] || []).concat(timestampsB[d] || [])).size / numTimestamps
        }))

    // Filter node IDs by quality.
    const nodes = new Set(nodeQualities.filter(d => d.quality >= quality).map(d => d.id))
    return data.filter(d => nodes.has(d['user_a']) && nodes.has(d['user_b']))
}

function removeExternalDevices(data) {
    log.i('removing external devices')
    return data.filter(d => +d['user_b'] !== -2)
}

function removeEmptyScans(data) {
    log.i('removing empty scans')
    return data.filter(d => +d['user_b'] !== -1)
}

function filterByRssi(data, rssi) {
    log.i('filtering by RSSI')
    return data.filter(d => d['rssi'] >= rssi)
}

function mapColumns(data) {
    log.i('mapping node IDs')

    // Create mapping from old to new IDs.
    // Collect IDs from both ends.
    const ids = [...new Set([
        ...new Set(data.map(d => +d['user_a'])),
        ...new Set(data.map(d => +d['user_b']))
    ])]

        // Sort IDs.
        .sort((a, b) => a - b)

        // Map IDs.
        .reduce((map, d, i) => {
            map[d] = i
            return map
        }, {})

    // Return mapped columns.
    return data.map(d => ({
        timestamp: +d['timestamp'] / 300,
        n1: ids[+d['user_a']],
        n2: ids[+d['user_b']]
    }))
}

function calculateMetrics(data) {
    log.i('calculating metrics:')
    log.i(`  number of nodes: ${new Set([...new Set(data.map(d => d.n1)), ...new Set(data.map(d => d.n2))]).size}`)
    log.i(`  number of links: ${data.length}`)
    log.i(`  number of bins:  ${new Set(data.map(d => d.timestamp)).size}`)
}

// Read arguments.
const parser = new ArgumentParser({
    version: '0.0.1',
    addHelp: true,
    description: 'Tool to prepare bluetooth data for simulations.'
})
parser.addArgument(['--input'], {
    dest: 'input',
    required: true,
    help: 'Input file path containing the raw data.'
})
parser.addArgument(['--output'], {
    dest: 'output',
    required: true,
    help: 'Output file path to save processed data to.'
})
parser.addArgument(['--remove-external-devices'], {
    dest: 'removeExternalDevices',
    action: 'storeTrue',
    help: 'Remove external devices [false].'
})
parser.addArgument(['--remove-empty-scans'], {
    dest: 'removeEmptyScans',
    action: 'storeTrue',
    help: 'Remove empty scans [false].'
})
parser.addArgument(['--quality'], {
    dest: 'quality',
    type: 'float',
    defaultValue: 0,
    help: 'Minimum quality to keep a node in the data set [0].'
})
parser.addArgument(['--rssi'], {
    dest: 'rssi',
    type: 'int',
    defaultValue: -101,
    help: 'Minimum RSSI value to accept a link [-101].'
})
const args = parser.parseArgs();


// Run main.
(async () => {
    // Load raw data.
    // Description of the data: https://www.nature.com/articles/s41597-019-0325-x/tables/2.
    log.i('reading raw data')
    let data = await io.readCSV(args.input)

    // Filter by quality. For the quality filter, we consider all scans (even empty ones).
    // Quality is defined by the fraction of timestamps with a scan.
    if (args.quality > 0) {
        data = filterByQuality(data, args.quality)
    }

    // Remove external devices.
    if (args.removeExternalDevices) {
        data = removeExternalDevices(data)
    }

    // Remove empty scans.
    if (args.removeEmptyScans) {
        data = removeEmptyScans(data)
    }

    // Filter by RSSI.
    if (args.rssi > -101) {
        data = filterByRssi(data, args.rssi)
    }

    // Convert data:
    // - Reduce timestamps to indices.
    // - Map old node IDs to a reduced set of consecutive IDs.
    data = mapColumns(data)

    // Calculate metrics.
    calculateMetrics(data)

    // Save processed data.
    await io.writeCSV(data, args.output)
})()
