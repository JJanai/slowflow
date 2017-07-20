#include <iostream>
#include <stdio.h>
#include <string.h>
#include "parameter_list.h"

using namespace std;


ParameterList::ParameterList() :
	verbose(string(max_verbosity_states,'0')), file(""), sequence_start(0), output(""), F(0), Jets(0),
	file_gt(""), center(Point(-1,-1)), extent(Point(-1,-1)),
	bf_weights(vector<double>(0)), exps(1), current_exp(0) {

}

ParameterList::ParameterList(string filename) :
	verbose(string(max_verbosity_states,'0')), file(""), sequence_start(0), output(""), F(0), Jets(0),
    file_gt(""), center(Point(-1,-1)), extent(Point(-1,-1)),
    bf_weights(vector<double>(0)), exps(1), current_exp(0)
{
	read(filename);
}

ParameterList::ParameterList(ParameterList& params) :
	verbose(params.verbose), file(params.file), file_list(params.file_list), id_list(params.id_list), name_list(params.name_list), category_list(params.category_list), sequence_start(params.sequence_start), sequence_start_list(params.sequence_start_list),
	jet_estimation(params.jet_estimation), jet_weight(params.jet_weight), jet_fps(params.jet_fps), jet_S(params.jet_S), output(params.output), F(params.F), Jets(params.Jets),
    file_gt(params.file_gt), file_gt_list(params.file_gt_list), occlusions_list(params.occlusions_list), pixel(params.pixel), set(params.set), center(params.center), extent(params.extent),
    bf_weights(params.bf_weights),
	insert_order_it(params.insert_order_it), params(params.params), paramslist(params.paramslist), paramslist_iterator(params.paramslist_iterator),
	exps(params.exps), current_exp(params.current_exp)
{
}

void ParameterList::read(string filename) {
    // read in config from file
    FILE* f = fopen(filename.c_str(), "rb");
    if (f == NULL)
        perror("Error opening file");
    else
    {
        char buffer[200];
        while (!feof(f))
        {
            if(fgets(buffer, 200, f) == NULL)
                break;

            char* name = strtok(buffer, "\n");          // get rid of '\n' at the end
            name = strtok(name, "\t");                  // first token
            if (name != NULL && name[0] != '#') {
                char* value = strtok(NULL, "\t");           // next token
                if(value != 0) {
                    // get sequence file name
                    if(strcmp(name,"id") == 0) {
                    	id_list.push_back(atoi(value));

                        continue;
                    }

                    // get sequence file name
                    if(strcmp(name,"name") == 0) {
                        name_list.push_back(value);

                        continue;
                    }

                    // get sequence file name
                    if(strcmp(name,"category") == 0) {
                    	category_list.push_back(value);

                        continue;
                    }

                    // get sequence file name
                    if(strcmp(name,"file") == 0) {
                        file = value;
                        file_list.push_back(value);

                        continue;
                    }

                    // get sequence groundtruth file name
                    if(strcmp(name,"file_gt") == 0) {
                        file_gt = value;
                        file_gt_list.push_back(value);

                        continue;
                    }

                    // get sequence groundtruth file name
                    if(strcmp(name,"occlusions") == 0) {
                        occlusions_list.push_back(value);

                        continue;
                    }

                    // get output path for results
                    if(strcmp(name,"output") == 0) {
                    	output = value;

                        continue;
                    }

                    // get first frame number
                    if(strcmp(name,"start") == 0) {
                        sequence_start = atoi(value);

                        sequence_start_list.push_back(sequence_start);

                        continue;
                    }

                    // get jet estimation path
                    if(strcmp(name,"jet_estimation") == 0) {
                        jet_estimation.push_back(value);
                    }

                    // get jet weight path
                    if(strcmp(name,"jet_weight") == 0) {
                    	jet_weight.push_back(atof(value));
                    }

                    // get jet estimation path
                    if(strcmp(name,"jet_fps") == 0) {
                    	jet_fps.push_back(atoi(value));
                    }

                    // get jet estimation path
                    if(strcmp(name,"jet_S") == 0) {
                    	jet_S.push_back(atoi(value));
                    }

                    // get sequence length
                    if(strcmp(name,"F") == 0) {
                        F = atoi(value);

                        pixel = vector<vector<Point2f> >(F, vector<Point2f>(0));
                        set = vector<int>(F, 0);

                        continue;
                    }

                    // get sequence length
                    if(strcmp(name,"Jets") == 0) {
                    	Jets = atoi(value);

                        continue;
                    }

                    // get selected pixels
                    if(strcmp(name,"pixel") == 0) {
                        int i = atoi(value);   // get number of img
                        //i -= sequence_start;
                        i -= 1;
                        if(i < 0 || i >= (int) F)
                            continue;

                        char* coord = strtok(NULL, "\t");   // next token

                        while(coord != 0) {
                            char* x = strsep(&coord, ",");   // split x and y
                            char* y = strsep(&coord, ",");    // next token

                            if(x != 0 && y != 0) {
                                pixel[i].push_back(Point2f(atof(x),atof(y)));
                                set[i]++;
                            }

                            coord = strtok(NULL, "\t");   // next token
                        }

                        continue;
                    }

                    // get center for cropping
                    if(strcmp(name,"center") == 0) {
                        char* x = strtok(value, ",");   // split x and y
                        char* y = strtok(NULL, ",");    // next token

                        if(x != 0 && y != 0)
                            center = Point(atof(x),atof(y));

                        continue;
                    }

                    // get frame size for cropping
                    if(strcmp(name,"extent") == 0) {
                        char* x = strtok(value, ",");   // split x and y
                        char* y = strtok(NULL, ",");    // next token

                        if(x != 0 && y != 0)
                        	extent = Point(atof(x),atof(y));

                        continue;
                    }

                    // get bilateral filter weights
                    if(strcmp(name,"bf_weight") == 0) {
                        int i = atoi(value);   // get number of weight
                        if(i > 0) i--;
                        char* weight = strtok(NULL, "\t");   // next token

                        if(weight != 0) {
                            bf_weights.resize(i+1, 0);
                            bf_weights[i] = atof(weight);
                        }

                        continue;
                    }

                    if(strcmp(name,"verbose") == 0) {
                    	verbose = value;
                    	// do not continue for backward compatibility
                    }

                    // get any other parameter
                    string n = name;
                    insert(n, parse(value), true);
                } else {
                	if(name[0] != '\0' && name[0] != '#')
                		cerr << "Value to parameter '" << name << "' is missing!" << endl;
                }
            }
        }
        fclose (f);
    }

    for(uint32_t i = id_list.size(); i < name_list.size(); i++)
    	id_list.push_back(i);
}

void ParameterList::insert(string param, string val, bool overwrite) {
	vector<string> vals;
	vals.push_back(val);

	insert(param, vals, overwrite);
}

void ParameterList::insert(string param, vector<string> vals, bool overwrite) {
    map<string, uint32_t>::iterator it = params.find(param);
    if(it != params.end()) {
        // parameter already exists
        uint32_t idx = it->second;

        // remove old size of parameterlist from exps
        exps /= paramslist[idx].size();

        // overwrite parameter list
        if(overwrite)
            paramslist[idx] = vals;
        else
            paramslist[idx].insert(paramslist[idx].end(), vals.begin(), vals.end());

        // add new size of parameterlist from exps
        exps *= paramslist[idx].size();
    } else {
        // parameter does not exist
        uint32_t idx = paramslist.size();

        params.insert(pair<string,uint32_t>(param, idx));   // add parameter
        insert_order_it.push_back(params.find(param));      // store iterator
        paramslist.push_back(vals);                                 // store parameter values
        paramslist_iterator.push_back(0);                           // selected value

        // add parameterlist to exps
        exps *= paramslist[idx].size();
    }
}


uint32_t ParameterList::experiment(){
    return current_exp;
}

uint32_t ParameterList::experiments() {
    return exps;
}

string ParameterList::experimentName() {
	stringstream os;
	bool empty = true;
	// iterate over all parameters
	for(uint32_t i = 0; i < insert_order_it.size(); i++) {
		string name = insert_order_it[i]->first;
		uint32_t idx = insert_order_it[i]->second;
		uint32_t iter = paramslist_iterator[idx];

		// skip parameters with only one value
		if(paramslist[idx].size() == 1)
			continue;

		if(!empty) os << "_";

		os << name << "_" << paramslist[idx][iter];
		empty = false;
	}

	return os.str();
}

string ParameterList::currentParametersName() {
	stringstream os;
	// iterate over all parameters
	for(uint32_t i = 0; i < insert_order_it.size(); i++) {
		string name = insert_order_it[i]->first;
		uint32_t idx = insert_order_it[i]->second;

		// skip parameters with only one value
		if(paramslist[idx].size() == 1)
			continue;

		os << name << "\t";
	}

	return os.str();
}

string ParameterList::currentParametersTabs() {
	stringstream os;
	// iterate over all parameters
	for(uint32_t i = 0; i < insert_order_it.size(); i++) {
		uint32_t idx = insert_order_it[i]->second;

		// skip parameters with only one value
		if(paramslist[idx].size() == 1)
			continue;

		os << "\t";
	}

	return os.str();
}

string ParameterList::currentParametersValue() {
	stringstream os;
	// iterate over all parameters
	for(uint32_t i = 0; i < insert_order_it.size(); i++) {
		string name = insert_order_it[i]->first;
		uint32_t idx = insert_order_it[i]->second;
		uint32_t iter = paramslist_iterator[idx];

		// skip parameters with only one value
		if(paramslist[idx].size() == 1)
			continue;

		os << paramslist[idx][iter] << "\t";
	}

	return os.str();
}

void ParameterList::reset() {
    for(uint32_t i = 0; i < paramslist.size(); i++)
        paramslist_iterator[i] = 0;

    current_exp = 0;
}

bool ParameterList::hasNextExp() {
    return current_exp < (exps - 1);
}

bool ParameterList::nextExp() {
    if(!hasNextExp())
        return false;

    // increment iterator of each list
    for(uint32_t i = 0; i < paramslist.size(); i++) {
		// skip parameters with only one value
        if(paramslist[i].size() == 1)
            continue;

        paramslist_iterator[i]++;
        // decide whether to increment
        if(paramslist_iterator[i] == paramslist[i].size())
            paramslist_iterator[i] = 0;
        else
            break;
    }

    current_exp++;

    return true;
}

ostream& operator<<(ostream& os, const ParameterList& params) {
	os << params.cfgString();
	return os;
}

string ParameterList::cfgString(bool all_exps) const {
	stringstream os;
	// compute the maximal length of a parameter name to align all with tabs
	int max_length = 0;
	for(uint32_t i = 0; i < insert_order_it.size(); i++) {
		string name = insert_order_it[i]->first;
		max_length = max(max_length, (int) name.length());
	}

	int tabs = ceil(max_length/8) + 1;
//	int tabs = 1;
	for(uint32_t f = 0; f < id_list.size(); f++)
		os << "id" << string(tabs,'\t') << id_list[f] << "\n";

	for(uint32_t f = 0; f < name_list.size(); f++)
		os << "name" << string(tabs,'\t') << name_list[f] << "\n";

	for(uint32_t f = 0; f < file_list.size(); f++)
		os << "file" << string(tabs,'\t') << file_list[f] << "\n";
	if(file_list.size() == 0)
		os << "file" << string(tabs,'\t') << file << endl;

	for(uint32_t f = 0; f < category_list.size(); f++)
		os << "category" << string(tabs,'\t') << category_list[f] << "\n";

	for(uint32_t f = 0; f < file_gt_list.size(); f++)
		os	 << "file_gt" << string(tabs,'\t') << file_gt_list[f] << "\n";
	if(file_gt_list.size() == 0)
		os << "file_gt" << string(tabs,'\t') << file_gt << endl;

	for(uint32_t f = 0; f < occlusions_list.size(); f++)
		os	 << "occlusions" << string(tabs,'\t') << occlusions_list[f] << "\n";

	if(!output.empty())
		os	<< "output" << string(tabs,'\t') << output << "\n" << "\n";

	for(uint32_t f = 0; f < sequence_start_list.size(); f++)
		os	<< "start" << string(tabs,'\t') << sequence_start_list[f] << "\n";
	if(sequence_start_list.size() == 0)
		os << "start" << string(tabs,'\t') << sequence_start << endl;

	for(uint32_t f = 0; f < jet_estimation.size(); f++)
		os << "jet_estimation" << string(tabs,'\t') << jet_estimation[f] << "\n";

	for(uint32_t f = 0; f < jet_weight.size(); f++)
		os << "jet_weight" << string(tabs,'\t') << jet_weight[f] << "\n";

	for(uint32_t f = 0; f < jet_fps.size(); f++)
		os << "jet_fps" << string(tabs,'\t') << jet_fps[f] << "\n";

	for(uint32_t f = 0; f < jet_S.size(); f++)
		os << "jet_S" << string(tabs,'\t') << jet_S[f] << "\n";

	os	<< "F" << string(tabs,'\t') << F << "\n";
	os	<< "Jets" << string(tabs,'\t') << Jets << "\n";

	for(uint32_t i = 0; i < F; i++) {
		if(i < set.size() && set[i] > 0) {
//			os << "pixels " << i << string(ceil(max_length/8),'\t') ;
			os << "pixels " << i << "\t";

			for(int p = 0 ; p < set[i]; p++)
				os << pixel[i][p] << "\t";

			os << "\n";
		}
	}

	if(extent.x > 0 || extent.y > 0)
		os << "extent" << string(tabs,'\t') << extent  << "\n";
	if(center.x > 0 || center.y > 0)
		os << "center" << string(tabs,'\t') << center  << "\n";

	os << "\n";

	// iterate over all other parameters
	for(uint32_t i = 0; i < insert_order_it.size(); i++) {
		string name = insert_order_it[i]->first;
		uint32_t idx = insert_order_it[i]->second;
		uint32_t iter = paramslist_iterator[idx];

		// print parameter name and add tabs
//		int tabs = ceil(max_length/8) - floor(name.length()/8) + 1;
//		os << name << string(tabs,'\t');
		os << name << string(1,'\t');

		if(!all_exps || paramslist[idx].size() == 1)
			os << paramslist[idx][iter];

		// print parameter list
		if(paramslist[idx].size() > 1) {
			if(!all_exps) os << "\t # in ";
			os << "(";
			for(uint32_t i = 0; i < paramslist[idx].size(); i++) {
				if(i > 0) os << ",";
				os << paramslist[idx][i];
			}
			os << ")";
		}
		os << "\n";
	}

	if(bf_weights.size() > 0) {
//		os << "bf_weights" << string(ceil(max_length/8),'\t') ;
		os << "bf_weights\t";
		for(uint32_t i = 0; i < bf_weights.size(); i++) {
			os << bf_weights[i] << "\t";
		}
		os << "\n";
	}

	return os.str();
}

vector<string> ParameterList::parse(char* value) {
    vector<string> val_list;

    // check if list of paramter values specified
    if(value[0] == '(') {
        // copy for strok
        char* cpy = new char[strlen(value)+1];
        strcpy(cpy,value);

        char* list = strtok(cpy, "(");                  // get rid of '('
        list = strtok(list, ")");                       // get rid of ')'
        list = strtok(list, ",");                       // iterate over list
        while(list != NULL) {
            val_list.push_back(list);
            list = strtok(NULL, ",");      // iterate over set
        }

        // delete copy
        delete[] cpy;
    } else {
        val_list.push_back(value);
    }

    return val_list;
}

void ParameterList::print() {
    int max_length = 0;
    for(uint32_t i = 0; i < insert_order_it.size(); i++) {
        string name = insert_order_it[i]->first;
        max_length = max(max_length, (int) name.length());
    }

    int tabs = ceil(max_length/8) + 1;
    if(current_exp == 0) {
    	for(uint32_t f = 0; f < id_list.size(); f++)
    		cout << "id" << string(tabs,'\t') << id_list[f] << endl;

    	for(uint32_t f = 0; f < name_list.size(); f++)
    		cout << "name" << string(tabs,'\t') << name_list[f] << endl;

    	for(uint32_t f = 0; f < file_list.size(); f++)
    		cout << "file" << string(tabs,'\t') << file_list[f] << endl;
    	if(file_list.size() == 0)
    		cout << "file" << string(tabs,'\t') << file << endl;

    	for(uint32_t f = 0; f < category_list.size(); f++)
    		cout << "category" << string(tabs,'\t') << category_list[f] << endl;

    	for(uint32_t f = 0; f < file_gt_list.size(); f++)
    		cout << "file_gt" << string(tabs,'\t') << file_gt_list[f] << endl;
    	if(file_gt_list.size() == 0)
    		cout << "file_gt" << string(tabs,'\t') << file_gt << endl;

    	for(uint32_t f = 0; f < occlusions_list.size(); f++)
    		cout << "occlusions" << string(tabs,'\t') << occlusions_list[f] << endl;

        cout << "output" << string(tabs,'\t') << output << endl << endl;

    	for(uint32_t f = 0; f < sequence_start_list.size(); f++)
    		cout << "start" << string(tabs,'\t') << sequence_start_list[f] << endl;
    	if(sequence_start_list.size() == 0)
    		cout << "start" << string(tabs,'\t') << sequence_start << endl;

    	for(uint32_t f = 0; f < jet_estimation.size(); f++)
    		cout << "jet_estimation" << string(tabs,'\t') << jet_estimation[f] << "\n";

    	for(uint32_t f = 0; f < jet_weight.size(); f++)
    		cout << "jet_weight" << string(tabs,'\t') << jet_weight[f] << "\n";

    	for(uint32_t f = 0; f < jet_fps.size(); f++)
    		cout << "jet_fps" << string(tabs,'\t') << jet_fps[f] << "\n";

    	for(uint32_t f = 0; f < jet_S.size(); f++)
    		cout << "jet_S" << string(tabs,'\t') << jet_S[f] << "\n";

        cout << "F" << string(tabs,'\t') << F << endl;
        cout << "Jets" << string(tabs,'\t') << Jets << endl;

        for(uint32_t i = 0; i < F; i++) {
            if(i < set.size() && set[i] > 0) {
                cout << "pixels " << i << string(ceil(max_length/8),'\t');

                for(int p = 0 ; p < set[i]; p++)
                    cout << pixel[i][p] << "\t";

                cout << endl;
            }
        }

        if(extent.x > 0 || extent.y > 0)
            cout << "extent" << string(tabs,'\t') << extent  << endl;
        if(center.x > 0 || center.y > 0)
            cout << "center" << string(tabs,'\t') << center  << endl;
    }

    cout << endl;
    if(exps > 1)
        cout << "--------------------- Experiment " << (current_exp + 1) << " of " << exps << " ---------------------" << endl;

    cout << "Parameters set to:" << endl;
    for(uint32_t i = 0; i < insert_order_it.size(); i++) {
        string name = insert_order_it[i]->first;
        uint32_t idx = insert_order_it[i]->second;
    	uint32_t iter = paramslist_iterator[idx];

        // print parameter name and add tabs
        int tabs = ceil(max_length/8) - floor(name.length()/8) + 1;
        cout << "\t" << name << string(tabs,'\t') << paramslist[idx][iter];

        // print parameter list
    	if(paramslist[idx].size() > 1) {
            cout << "\t in (";
    		for(uint32_t i = 0; i < paramslist[idx].size(); i++) {
    			if(i > 0)
    				cout << ",";
    			cout << paramslist[idx][i];
    		}
            cout << ")";
    	}
		cout << endl;
    }

    if(bf_weights.size() > 0) {
        cout << "\tbf_weights" << string(ceil(max_length/8),'\t');
        for(uint32_t i = 0; i < bf_weights.size(); i++) {
            cout << bf_weights[i] << "\t";
        }
        cout << endl;
    }
}

bool ParameterList::exists(string param) {
    return params.find(param) != params.end();
}

void ParameterList::setParameter(string param, string value) {
    if(!exists(param)) {
        vector<string> vals;
        vals.push_back(value);
        insert(param,vals);
    }

    uint32_t idx = params[param];
    uint32_t it = paramslist_iterator[idx];
    paramslist[idx][it] = value;
}

string ParameterList::parameter(const char* param) {
    if(!exists(param)) {
        cerr << "Error: Parameter " << param << " does not exist!" << endl;
        return "";
    }

	uint32_t idx = params[param];
	uint32_t it = paramslist_iterator[idx];
	return paramslist[idx][it];
}

template<>
string ParameterList::parameter<string>(string param, string def) {
    if(!exists(param)) {
        return def;
    }

	uint32_t idx = params[param];
	uint32_t it = paramslist_iterator[idx];
	return paramslist[idx][it];
}

template<>
int ParameterList::parameter<int>(string param, string def) {
    if(!exists(param)) {
        if(!def.empty()) return atoi(def.c_str());

        cerr << "Error: Parameter " << param << " does not exist!" << endl;
        return 0;
    }

	uint32_t idx = params[param];
	uint32_t it = paramslist_iterator[idx];
	return atoi(paramslist[idx][it].c_str());
}

template<>
double ParameterList::parameter<double>(string param, string def) {
    if(!exists(param)) {
        if(!def.empty()) return atof(def.c_str());

        cerr << "Error: Parameter " << param << " does not exist!" << endl;
        return 0;
    }

	uint32_t idx = params[param];
	uint32_t it = paramslist_iterator[idx];
	return atof(paramslist[idx][it].c_str());
}

template<>
float ParameterList::parameter<float>(string param, string def) {
    if(!exists(param)) {
        if(!def.empty()) return atof(def.c_str());

        cerr << "Error: Parameter " << param << " does not exist!" << endl;
        return 0;
    }

	uint32_t idx = params[param];
	uint32_t it = paramslist_iterator[idx];
	return atof(paramslist[idx][it].c_str());
}

template<>
bool ParameterList::parameter<bool>(string param, string def) {
    if(!exists(param)) {
    	 if(!def.empty()) return (def == "0") ? false : true;

        cerr << "Error: Parameter " << param << " does not exist!" << endl;
        return false;
    }

	uint32_t idx = params[param];
	uint32_t it = paramslist_iterator[idx];
	return (paramslist[idx][it] == "0") ? false : true;
}

template<>
vector<int> ParameterList::splitParameter<int>(string param, string def) {
	char* split_str = NULL;

    if(!exists(param)) {
		if(def.length() > 0) {
			split_str = new char[def.length() + 1];
			strcpy(split_str, def.c_str());
		}
    } else {
		uint32_t idx = params[param];
		uint32_t it = paramslist_iterator[idx];

		if(paramslist[idx][it].length() > 0) {
			split_str = new char[paramslist[idx][it].length() + 1];
			strcpy(split_str, paramslist[idx][it].c_str());
		}
    }

    vector<int> output;

    if(split_str != NULL) {
		char* part = strtok(split_str, ",");   // next token
		while (part != NULL) {
			output.push_back(atoi(part));
			part = strtok(NULL, ",");   // next token
		}

		delete[] split_str;
    }

    return output;
}

template<>
vector<float> ParameterList::splitParameter<float>(string param, string def) {
	char* split_str = NULL;

    if(!exists(param)) {
		if(def.length() > 0) {
			split_str = new char[def.length() + 1];
			strcpy(split_str, def.c_str());
		}
    } else {
		uint32_t idx = params[param];
		uint32_t it = paramslist_iterator[idx];

		if(paramslist[idx][it].length() > 0) {
			split_str = new char[paramslist[idx][it].length() + 1];
			strcpy(split_str, paramslist[idx][it].c_str());
		}
    }

    vector<float> output;

    if(split_str != NULL) {
		char* part = strtok(split_str, ",");   // next token
		while (part != NULL) {
			output.push_back(atof(part));
			part = strtok(NULL, ",");   // next token
		}

		delete[] split_str;
    }

    return output;
}

template<>
int ParameterList::maximum(string param){
    if(!exists(param))
        throw std::logic_error( "Parameter used to obtain minimum does not exist!" );

	uint32_t idx = params[param];
	int maxi = 0;
	for(uint32_t it = 0; it < paramslist[idx].size(); it++)
		maxi = std::max(maxi, atoi(paramslist[idx][it].c_str()));

	return maxi;
}


template<>
int ParameterList::minimum(string param){
    if(!exists(param))
        throw std::logic_error( "Parameter used to obtain minimum does not exist!" );

	uint32_t idx = params[param];
	int mini = 0;
	for(uint32_t it = 0; it < paramslist[idx].size(); it++)
		mini = std::min(mini, atoi(paramslist[idx][it].c_str()));

	return mini;
}
