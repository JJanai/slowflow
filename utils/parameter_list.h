#ifndef PARAMETERLIST_H
#define PARAMETERLIST_H

#include <string>
#include <sstream>
#include <vector>
#include <map>

#include <iostream>

#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>

using namespace std;
using namespace cv;

const int max_verbosity_states =10;
enum Verbosity {VER_CMD=0, VER_IN_GT=1, VER_IMG_PYR=2, VER_FLO_PYR=3, WRITE_FILES=4};

class ParameterList
{
public:
    ParameterList();
    ParameterList(string file);
    ParameterList(ParameterList& params);		// deep clone

    void read(string file);

    // set parameters
    template<typename T>
    void setParameter(string param, T value) {
        stringstream v;
        v << value;
        setParameter(param, v.str());
    }
    void setParameter(string param, string value);

    // insert a parameter param with values vals
    void insert(string param, string val, bool overwrite=false);
    void insert(string param, vector<string> vals, bool overwrite=false);

    // check existence
    bool exists(string param);

    // print all parameters
    void print();

    // iterate over all experiments
    string experimentName();
    string currentParametersName();
    string currentParametersTabs();
    string currentParametersValue();
    uint32_t experiment();
    uint32_t experiments();
    void reset();
    bool hasNextExp();
    bool nextExp();

    void advance(uint32_t exp) {
    	while(experiment() < exp && hasNextExp())
    		nextExp();
    }

    // get a binary code
    inline bool verbosity(uint32_t state) {
    	if(state < verbose.size() && verbose[state] == '1')
    		return true;
    	else
    		return false;
    }

    // get a parameter value with type T
    string parameter(const char* param);
    template<typename T>
    T parameter(string param) { return parameter<T>(param, "");  }
    template<typename T>
    T parameter(string param, string def);

    // get a vector of parameter values with type T
    template<typename T>
    vector<T> splitParameter(string param) { return parameter<T>(param, "");  }
    template<typename T>
    vector<T> splitParameter(string param, string def);

    // get max value
    template<typename T>
    T maximum(string param);
    template<typename T>
    T minimum(string param);

    int id(int s) {
    	if(s < (int) id_list.size())
    		return id_list[s];
    	else
    		return s;
    }

    // output
    string cfgString(bool all_exps = false) const;
    friend ostream& operator<<(ostream& os, const ParameterList& test);

    // name parameters
    string verbose;

    // sequence
    string file;
    vector<string> file_list;
    vector<int> id_list;
    vector<string> name_list;
    vector<string> category_list;
    u_int32_t sequence_start;						// first image in sequence
    vector<u_int32_t> sequence_start_list;			// first image in sequence
    vector<string> jet_estimation;
    vector<double> jet_weight;
    vector<int> jet_fps;
    vector<int> jet_S;
    string output;
    u_int32_t F;									// number of frames
    u_int32_t Jets;									// number of jets

    // ground truth
    string file_gt;
    vector<string> file_gt_list;
    vector<string> occlusions_list;
    vector<vector<Point2f> > pixel;
    vector<int> set;
    Point center;
    Point extent;

    // bilateral weights
    vector<double> bf_weights;
private:
    vector<string> parse(char* value);

    // any scalar parameter
    vector<map<string, uint32_t>::iterator > insert_order_it;
    map<string, uint32_t> params;
    vector<vector<string> > paramslist;
    vector<uint32_t > paramslist_iterator;

    int exps;
    int current_exp;
};

#endif // PARAMETERLIST_H
