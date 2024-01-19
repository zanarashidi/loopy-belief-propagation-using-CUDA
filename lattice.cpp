#include <iostream>
#include <fstream>

using namespace std;

/*
 * This code produces 2D square lattice factor graphs.
 * First it generates the number of factors and variables in one line, 
 * then it writes into the text file, the number of neighbors for each 
 * factor and their indexes in each line.
 * 
 * For example: A 2*2 Lattice would be:
 * 4 4
 * 2 0 1
 * 2 0 3
 * 2 1 2
 * 2 2 3
 */ 

int main(){
	int side;
	cout << "Enter Grid Dim: ";
	cin >> side;
	int vn = side*side;
	int fn = 2*(side*(side-1));
	
	ofstream outfile("lattice.txt");
	
	outfile << vn << " " << fn << endl;
	for(int j=0; j<side; j++){
		for(int k=0; k<side-1; k++){
				outfile << "2 " << k+j*side << " " << k+j*side+1 << endl;
		}
		if(j==side-1) break;
		for(int l=0; l<side; l++){
			outfile << "2 " << l+j*side << " " << l+(j+1)*side << endl;
		}
	}
	
	outfile.close();
	
	return 0;
}
