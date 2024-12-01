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

int main()
{

    bool proc = false;
    if (proc){
        ifstream ifile("esetek2_3.txt");
        ofstream ofile("esetek2_3+20.txt");
        int k=0;
        while (!ifile.eof()){
                k++;
            if (k%10000==0) cout<<k<<endl;
            float a,b,c,d,e;
            string s;
            ifile>>s>>a>>b>>c>>d>>e;
            if (a>=20){
                ofile<<s<<" "<<a<<" "<<b<<" "<<c<<" "<<d<<" "<<e<<endl;
            }
        }
        return 0;
    }


    bool histo = true;
    if (true){
        ifstream ifile("esetek1_3+20.txt");
        ofstream ofile("esetek1_3+20_histo.txt");
        vector<float> csoportok(100,0);
        int k=0;
        while (!ifile.eof()){
                k++;
            if (k%10000==0) cout<<k<<endl;
            float a,b,c,d,e;
            string s;
            ifile>>s>>a>>b>>c>>d>>e;
            if (a>=20){
                float f=c*100; int g=floor(f);
                csoportok[g]+=a;
                //ofile<<s<<" "<<a<<" "<<b<<" "<<c<<" "<<d<<" "<<e<<endl;
            }
        }
        for (int i=0; i<100; i++){
            ofile<<csoportok[i]<<endl;
        }
        return 0;
    }

    return 0;
}

























