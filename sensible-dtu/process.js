import io from './src/io.js'
import { ArgumentParser } from 'argparse'


(async () => {
    const data = await io.readCSV('output/events.csv')
})()
