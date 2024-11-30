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
    ifstream ifile("esetek2.txt");
    ofstream ofile("esetek2_+20.txt");
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

























