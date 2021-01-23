/******************************************************/
/*                Laurent Hébert-Dufresne             */
/*               Percolate on an edgelist             */
/*                   Santa Fe Institute               */
/******************************************************/

#include <iostream>
#include <cmath>
#include <fstream>
#include <string>
#include <cstdlib>
#include <vector>
#include <sstream>
#include <iomanip>
#include <algorithm>
#include <numeric>
#include <stdio.h>
#include <time.h>
#include <set>
//elements of BOOST library (for random number generator)
  #include <boost/random/mersenne_twister.hpp>
  #include <boost/random/uniform_real.hpp>
  #include <boost/random/variate_generator.hpp>

//Random number generator
  boost::mt19937 generator(static_cast<unsigned int>(time(NULL))); //
  boost::uniform_real<> uni_dist(0,1);
  boost::variate_generator<boost::mt19937&, boost::uniform_real<> > uni(generator, uni_dist);

using namespace std;

void percolateandtrace(vector< vector<int> > &, vector<int> &, vector<int> &, double, double ); //function prototype

//MAIN
int main(int argc, const char *argv[])
{
  //ARGUMENTS
  std::string filename = argv[1]; //path to edge list
  double T = atof(argv[2]); //transmission probability
  double P = atof(argv[3]); //contact tracing probability
  double N = atoi(argv[4]); //number of simulations

//INPUT AND CONSTRUCTION OF STRUCTURES
//PREPARES INPUT & OUTPUT
ifstream input(filename.c_str());
string line;
ifstream input0(filename.c_str());
int temp = 0;
int MAX = 0;
int node1;
int node2;
vector<int> Degrees;
if (!input0.is_open()) {
	cerr<<"error in filenames of input directory!"<<endl;;
 	return EXIT_FAILURE;
} //end if
else {
	while ( input0 >> temp ) if(temp > MAX) MAX = temp;
}
MAX = MAX+1;
int link=0;
//cout << "Number of nodes: " << MAX << endl;

//reading network
int links=0;
if (!input.is_open()) {
	cerr<<"error in filenames of input directory!"<<endl;;
 	return EXIT_FAILURE;
} //end if
else {
	while ( input >> node1 >> std::ws >> node2 >> std::ws ) {
        if(input.eof()) break;
		else{Degrees.push_back(node1); Degrees.push_back(node2);}
	} //end while
} //end else
input.close();
if((Degrees.size() % 2)) Degrees.pop_back();

int nb_data_point = 0;
double sum_of_data_points = 0.0;

for(int net=0; net<N; ++net) {

    vector< vector<int> > adjmat; //[neighbours]
    adjmat.resize(MAX);

    for(int fun=0; fun<3; ++fun) random_shuffle(Degrees.begin(),Degrees.end());

    //randomize network
    for(int k=0; k<Degrees.size(); k+=2) {
        node1 = Degrees[k]; node2 = Degrees[k+1];
        adjmat[node1].push_back(node2);
        adjmat[node2].push_back(node1);
    }    

    for(int sim=0; sim<1000; ++sim) {

        //set up the epidemic
        vector<int> status(MAX,0);
        vector<int> newcases;
        int patientzero = int(floor(uni()*MAX));
        status[patientzero] = 1;
        newcases.push_back(patientzero);
    
        //run the epidemic
        percolateandtrace(adjmat, status, newcases, T, P);
    
        //calculate cluster size
        int largest_cluster_size = accumulate(status.begin(), status.end(), 0);

        if(1.0*largest_cluster_size>(0.01*MAX)) {
            sum_of_data_points += static_cast<double>(largest_cluster_size) / MAX;
            ++nb_data_point;
        }

    }

}//end for sims

//output
cout.precision(6);
if(nb_data_point > 0) cout << std::fixed << T << "  " << P << "  " << sum_of_data_points/nb_data_point << "  " << 1.0*nb_data_point/(1000.0*N) << endl;
else  cout << std::fixed << T << "  " << P << "  " << 0.0 << "  " << 0.0 << endl;

return 0;
} //end main

void percolateandtrace(vector< vector<int> > &neighbour, vector<int> &status, vector<int> &newcases, double T, double P)
	{ 
        vector<int> currentcases = newcases;
        newcases.clear();
        for(int cas=0; cas<currentcases.size(); ++cas) {
            int currentinfected = currentcases[cas];
            vector<int> possiblenewcases;
            bool trace = false;
            for(int voisin=0; voisin<neighbour[currentinfected].size(); ++voisin) {
                int currentneighbour = neighbour[currentinfected][voisin];
                if(status[currentneighbour]==0 && uni() < T) {
                    status[currentneighbour] = 1;
                    if(uni()>P && !trace) possiblenewcases.push_back(currentneighbour);
                    else {trace = true; possiblenewcases.clear();}
                }
            }
            percolateandtrace(neighbour, status, possiblenewcases, T, P);
        }

	}


