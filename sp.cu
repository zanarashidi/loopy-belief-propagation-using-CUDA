/*
 * Sum-Product Algorithm using GPU
 * 
 * Written by: Zana Rashidi 
 * 
 * As part of the B.Sc. Project in Computer Engineering
 * 
 * Computer Engineering Department, Sharif University of Technology
 * 
 * Supervisor: Mahdi Jafari Siavoshani
 * 
 * July 2017
 */

#include <iostream>
#include <cmath>
#include <fstream>
#include <string>
#include <sstream>
#include <ctime>
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "cuda.h"

#define max_neigh 4 //maximum number of neighbors for each node
#define eps 1e-6 //epsilon for tolerance
#define threads_pb 512 //threads per block
#define max_itrs 500 //maximum number of iterations

using namespace std;

class node;
class svf;

/*
 * Single Variable Function:
 * This is a function of one variable and used throughout the code to 
 * indicate marginals and messages between nodes which are both single
 * variable. In this case the variables have two states.
 * 
 * __host__ __device__:
 * Every function with this prefix can be called both from the cpu 
 * (host) and gpu (device).
 * 
 */

class svf
{	
	public:
		double x[2]; //two states
		__host__ __device__ svf() {
		}
		__host__ __device__ svf(double b0, double b1){
			x[0]=b0;
			x[1]=b1;
		}
		__host__ __device__ svf(const svf &obj){ //this is a copy constructor needed when copying the objects (svf) into the device
			for(int i=0; i<2; i++){
				x[i]=obj.x[i];
			}
		}
		__host__ __device__ void normalize() { //this is functions which normalizes (sum equal to one) an svf
			double t = 0;
			for(int i=0; i<2; i++){
				t += x[i];
			}
			for(int i=0; i<2; i++){
				x[i] /= t;
			}
		}
		void print() {
			cout << '(' << x[0] << ',' << x[1] << ") ";
		}
};

__host__ __device__ svf mult(svf a, svf b) { //this funtions multiplies two svfs by element
	svf s;
	for(int i=0; i<2; i++) {
		s.x[i] = a.x[i] * b.x[i];
	}
	return s;
}

__host__ __device__ svf add(svf a, svf b) { //this functions adds two svfs by element
	svf s;
	for(int i=0; i<2; i++) {
		s.x[i] = a.x[i] + b.x[i];
	}
	return s;
}

__host__ __device__ double compare_marginal(svf a, svf b){ //this function returns distance between two svfs 
	a.normalize();
	b.normalize();
	return (double)abs(a.x[0] - b.x[0]) + abs(a.x[1] - b.x[1]); 
}

/*
 * Node:
 * This class represents each node of the factor graph, variables and 
 * factors are indicated by a boolean. 
 */

class node
{
	public:
	bool is_var; //this bolean idicates if the node is a variable (or a factor)
	svf prv_marginal; //this is the saved previous marginal of it
	int my_ind; //this is its index
	int num_neighbors; //this is its number of neighbors
	int neighbors[max_neigh]; //this array contains its neighbors' indexes
	svf msgs[max_neigh]; //this array contains last recieved messages from its neighbors
	
	__host__ __device__ node() {
		num_neighbors = 0;
		prv_marginal = svf(double(1),double(1)); //initial marginal
	}
	
	__host__ __device__ node(const node &obj) { //this is a copy constructor needed when copying the objects (nodes) into the device
		is_var = obj.is_var;
		prv_marginal = obj.prv_marginal;
		my_ind = obj.my_ind;
		num_neighbors = obj.num_neighbors;
		for(int i=0; i<max_neigh; i++){
			neighbors[i] = obj.neighbors[i];
			msgs[i] = obj.msgs[i];
		}
	}

	__host__ __device__ svf marginal() { //this function calculates the node's marignal
		svf tmp = svf(1,1); //unit function
		for(int i=0; i<num_neighbors; i++) {
			tmp = mult(tmp, msgs[i]);
		}
		return tmp;
	}

	__host__ __device__ bool check_marginal(){ //this functions checks if the marginal has converged
		double diff = compare_marginal(marginal(), prv_marginal);
		prv_marginal = marginal();
		return (diff < eps);
	}

	__host__ __device__ void send_all(node *rec) { //this function calls the send function for every neighbor
		for(int i=0; i<num_neighbors; i++) {
			send(i, rec);
		}
	}

	__host__ __device__ void send_init(node *f) { //this function (called by variables) sends initial beliefs (unit) to all neighbors 
		for(int i=0; i<num_neighbors; i++) {
			f[neighbors[i]].get(my_ind, svf(double(1),double(1))); //initial marginal
		}
	}

	__host__ __device__ void send(int k, node *rec) { //this functions sends a new message to the k'th neighbor
		if(is_var) { //if this node (sender) is a variable
			svf tmp = svf(1,1); //unit funtion	
			if(num_neighbors==1){ //if the sender has only one neighbor (variable leaf in a tree) send initial marginal
				tmp = svf(double(1),double(1)); //initial marginal
			}
			else{ //otherwise send multiply of all incoming messages except the reciever
				for (int i=0; i<num_neighbors; i++) {
					if(i != k) {
						tmp = mult(tmp, msgs[i]);
					}
				}
			}
			rec[neighbors[k]].get(my_ind, tmp); //call the get function of the k'th neighbor
		} else { //if this node is a factor
			svf tmp = svf(0,0); //zero function
			for (int t=0; t<(1<<(num_neighbors-1)); t++) { //for all possible combinations of the variables in the factor except the reciever
				bool vars[max_neigh];
				int c=0;
				for(int i=0; i<num_neighbors; i++) { //first calculate the i'th digit of the combinations, where i is the index of the neighbor
					if(i!=k) {
						vars[i] = (bool) (t & (1<<c));
						c++;
					} else {
						vars[i] = 0;
					}
				}
				bool nd=1;
				double b=1;
				for(int i=0; i<num_neighbors; i++) { //then calculate the message result with that combination
					if(i!=k) {
						b = b * msgs[i].x[vars[i]]; //multiply the messages 
						nd = nd & vars[i]; //AND function
					}
				}
				svf s(b, b * !nd); //AND to NAND
				tmp = add(tmp, s); //sum the results to send 
			}
			rec[neighbors[k]].get(my_ind, tmp); //calling the get function on of the k'th neighbor
		}
	}

	__host__ __device__ void get(int n, svf m) { //this function recieves the message m from node n and saves it (normalized)
		int ind;
		for (int i=0; i<num_neighbors; i++) {
			if(neighbors[i] == n) {
				ind = i;
				break;
			}
		}
		m.normalize(); //normalize each at stage
		msgs[ind] = m;
	}
};

/*
 * GPU Kernel:
 * This function runs the sum-product algorithm in the gpu. __shared__ 
 * variables are shared in a block of threads. All of the calculations 
 * are in a single block. After each operations (of the threads:k) there
 * is a __syncthreads() function which makes sure untill all threads in
 * a block have finished their work. The algorithm is explained in the 
 * cpu section.
 */
__global__ void gpu_send_all(node *factors, node *variables, int num_f, int num_v){
	__shared__ int iterations; 
	__shared__ int t[threads_pb]; 
	__shared__ int temp;
	unsigned int k = threadIdx.x;
	iterations = 0;
	if(k < num_v){
		variables[k].send_init(factors);
	}
	__syncthreads();
	t[k] = 0;
	__syncthreads();
	while(iterations < max_itrs){
		temp = 1;
		if(k<num_f){
			factors[k].send_all(variables);
		}
		__syncthreads();
		if(k<num_v){
			variables[k].send_all(factors);
		}
		__syncthreads();
		if(k<num_v){
			t[k] = variables[k].check_marginal();
		}			
		__syncthreads();
		if(k<num_v){
			atomicAnd(&temp, t[k]); //atomicAnd performs the function 'and' on all the blocks on a thread andreduces them to one variable using atomic functions
		}
		__syncthreads();
		if(temp==1) break;
		iterations++;
	}
}

bool bit(long long n, int i) //this functions returns the i'th bit of the word n
{
	return (bool) (n & (1<<i));
}

/*
 * Brute Force:
 * This function calculates the marignals of the variables in a factor 
 * graph using brute force method which is checking all the possible 
 * combinations of the variables in the graph and calculating the 
 * possibility of each state of each variable regarding the functions.
 */ 

void bfmarginal(node *var, node *fac, int varn, int facn, svf *bfmarginals){
	long long valid_states = 0;
	long long *num_ones;
	num_ones = new long long[varn];
	for(int i=0; i<varn; i++){
		num_ones[i] = 0;
	}
	for(long long j=0; j<(1<<varn); j++){ //for all possible combination of the variables
		bool g_state=1;
		for(int k=0; k<facn; k++){ //first calculate the results of the factors on that combination
			bool f_state=1;
			for(int i=0; i<fac[k].num_neighbors; i++){
				f_state&=bit(j,fac[k].neighbors[i]);
			}
			g_state&=(~f_state);
		}
		if(g_state==1){ //then check if the result is valid and count the ones and zeros
			valid_states++;
			for(int i=0; i<varn; i++){
				num_ones[i]+=bit(j,i);
			}
		}
	}
	for(int j=0; j<varn; j++){ //finally devide the number of ones/zeros by all valid states
		double x1 = (double)num_ones[j]/valid_states;
		double x0 = (double)(valid_states-num_ones[j])/valid_states;
		bfmarginals[j] = svf(x0,x1);
	}
	delete[] num_ones;
}

int main()
{
	int vn = 0; 
	int fn = 0; 
	
	string line;
	ifstream infile("lattice.txt"); //get factor graph from file
		
	getline(infile, line);
	istringstream iss(line);
	iss >> vn >> fn; //first get the number of variables and factors
	
	node *v;
	node *f;
	
	v = new node[vn];
	f = new node[fn];
	
	for(int i=0; i<vn; i++){ //initilize my_ind and is_var
		v[i].my_ind = i;
		v[i].is_var = 1;
	}
	for(int i=0; i<fn; i++){
		f[i].my_ind = i;
		f[i].is_var = 0;
	}
	
	for(int i=0; i<fn; i++){ //then read the number of neighbors of each factor and their indexes
		getline(infile, line);
		istringstream iss(line);
		int k = 0;
		iss >> k; //number of neighbors of factor i
		for (int j=0; j<k; j++){
			int t = 0;
			iss >> t; //index of neighbors of factor i
			f[i].neighbors[j] = t;
			v[t].neighbors[v[t].num_neighbors] = i; //do the same for the variables
			v[t].num_neighbors++; //increase number of variables' neighors
		}
		f[i].num_neighbors = k;
	}
		
	infile.close();
	
	//copying the raw (variable & factor) nodes to use in cpu section
	
	node *vc;
	node *fc;
	
	vc = new node[vn];
	fc = new node[fn];
	
	for(int i=0; i<vn; i++){
		vc[i] = v[i];
	}
	for(int i=0; i<fn; i++){
		fc[i] = f[i];
	}		
	
	/*GPU Section*/ 
	
	cudaEvent_t start, stop;
	cudaEventCreate(&start);
	cudaEventCreate(&stop);
	cudaEventRecord(start, 0); //start gpu timer
	
	node *fg;
	node *vg;
	
	const size_t szf = size_t(fn) * sizeof(node); //size of factor and variable arrays
	const size_t szv = size_t(vn) * sizeof(node);

	cudaMalloc((void**)&fg, szf); //allocate space for this arrays on the device
	cudaMalloc((void**)&vg, szv);

	cudaMemcpy(fg, f, szf, cudaMemcpyHostToDevice); //copy the arrays into the device
	cudaMemcpy(vg, v, szv, cudaMemcpyHostToDevice);
	
	gpu_send_all <<<1, threads_pb>>>(fg, vg, fn, vn); //call the gpu kernel with one block and 512 threads per block
	
	cudaThreadSynchronize(); //wait for all the threads to finish their work
	
	cudaMemcpy(f, fg, szf, cudaMemcpyDeviceToHost); //copy back the information
	cudaMemcpy(v, vg, szv, cudaMemcpyDeviceToHost);
	
	cudaEventRecord(stop, 0); //stop gpu timer
	cudaEventSynchronize(stop);
	float elapsedTime;
	cudaEventElapsedTime(&elapsedTime, start, stop);

	cudaFree(vg); //free device memories
	cudaFree(fg);
	cudaEventDestroy(start);
	cudaEventDestroy(stop);
	
	/*CPU Section*/
		
	clock_t startCPU, stopCPU;
	startCPU = clock(); //start cpu timer
	
	for(int i=0; i<vn; i++){
		vc[i].send_init(fc); //send initial beliefs (marginals) from variables to factors
	}

	int iterations = 0; //iterator
	while(iterations < max_itrs){ //while less than maximum allowed iterations
		bool t = 1;
		for(int i=0; i<fn; i++){ //send messages from all factors to all variables
			fc[i].send_all(vc);
		}
		for(int i=0; i<vn; i++){ //send messages from all variables to all factors
			vc[i].send_all(fc);
		}
		for(int i=0; i<vn; i++){
			t &= vc[i].check_marginal(); //check the mariginals for the variables for convergency
		}
		
		if(t==1) break;
		iterations++;
	}
	
	stopCPU = clock(); //stop cpu timer
	
	/*Brute Force Section*/
	
	clock_t startBF, stopBF; 
	startBF = clock(); //start brute force timer
	
	svf *bfmarginals;
	bfmarginals = new svf[vn];
	
	bfmarginal(v, f, vn, fn, bfmarginals);	//call brute force method
	
	stopBF = clock(); //stop brute force timer
	
	//print the results
	
	cout << "Using " << iterations << " Iterations and " << eps << " Tolerance:" << endl;
	cout << "v[x]:(GPU) (CPU) (BrF)" << endl;
	cout << "Time:(" << elapsedTime << " ms) (" << (stopCPU-startCPU) << " ms) (" << (stopBF-startBF) << " ms)" << endl;
	for(int i=0; i<vn; i++) {
		svf s = v[i].marginal();
		s.normalize();
		cout << "v[" << i << "]:";
		s.print();
		s = vc[i].marginal();
		s.normalize();
		s.print();
		bfmarginals[i].print();
		cout << endl;
	}

	delete[] vc; //free memories allocated
	delete[] fc;
	delete[] v;
	delete[] f;
	delete[] bfmarginals;

	return 0;
}
