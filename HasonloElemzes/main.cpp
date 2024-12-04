#include <iostream>
#include <fstream>
#include <string>
#include <vector>
#include <unordered_set>
#include <windows.h>
#include <sstream>
#include <algorithm>
#include <set>
#include <thread>
#include <random>
#include <list>
#include <cmath>
#include <iostream>
#include <vector>
#include <thread>
#include <queue>
#include <mutex>
#include <condition_variable>
#include <functional>
#include <atomic>

using namespace std;

struct Par2{
    float ertek;
    float sorszam;

    bool operator<(const Par2& other) const {
        return ertek<other.ertek;
    }
};

int main()
{
    ifstream ifile("hasonlok_203_o.txt");

    vector<vector<float>> k1, k2, k3;
    while(!ifile.eof()){
        int zzz, s;
        ifile>>zzz>>s;
        cout<<zzz<<endl;
        string str;
        Par2 f1[s], f2[s], f3[s];
        for (int i=0; i<s; i++){
            ifile>>str>>str>>f1[i].ertek>>f1[i].sorszam>>str>>f2[i].ertek>>f2[i].sorszam>>str>>f3[i].ertek>>f3[i].sorszam;
        }
        int n = sizeof(f1)/sizeof(f1[0]);
        sort(f1,f1+n);
        sort(f2,f2+n);
        sort(f3,f3+n);
        vector<float> t1(s), t2(s), t3(s);
        for (int i=0; i<s; i++){
            t1[i]=f1[i].sorszam;
            t2[i]=f2[i].sorszam;
            t3[i]=f3[i].sorszam;
        }
        k1.push_back(t1); k2.push_back(t2); k3.push_back(t3);
    }

    vector<vector<float>> v;
    ofstream ofile("hasonlok_o3.txt");
    for (int i=0; i<k1.size(); i++){
        vector<float> temp;
        float ertekek[5]= {200,100,40,10,5};
        for (int j=0; j<5; j++){
            float szum = 0;
            for (int k=0; k<ertekek[j]; k++) szum+=k1[i][k];
            szum/=ertekek[j];
            temp.push_back(szum);
            ofile<<szum<<" ";
        }
        for (int j=0; j<5; j++){
            float szum = 0;
            for (int k=0; k<ertekek[j]; k++) szum+=k2[i][k];
            szum/=ertekek[j];
            temp.push_back(szum);
            ofile<<szum<<" ";
        }
        for (int j=0; j<5; j++){
            float szum = 0;
            for (int k=0; k<ertekek[j]; k++) szum+=k3[i][k];
            szum/=ertekek[j];
            temp.push_back(szum);
            ofile<<szum<<" ";
        }
        ofile<<endl;
        v.push_back(temp);

    }

    return 0;
}
