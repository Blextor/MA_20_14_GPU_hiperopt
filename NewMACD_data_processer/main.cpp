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

/// Függvény, amely felosztja a szöveget megadott karakterek mentén
std::vector<std::string> split(const std::string& text, const std::string& delimiters) {
    std::vector<std::string> tokens;
    std::string token;
    std::stringstream ss(text);

    while (std::getline(ss, token)) {
        std::size_t start = 0, end = 0;

        while ((end = text.find_first_of(delimiters, start)) != std::string::npos) {
            if (end != start) {
                tokens.push_back(text.substr(start, end - start));
            }
            start = end + 1;
        }

        if (start < text.size()) {
            tokens.push_back(text.substr(start));
        }
    }

    return tokens;
}

struct Parameterek{
    int adasVeteliNapok = 0; /// (0-5)
    int m1 = 1, m2=2, m3=3; /// a három vizsgált mozgóátlag (1-50, különözõek)
    int ms = 0; /// mozgóátlagok növekvõ sorrendje (0-5)
    int mi = 0; /// m1 és m3 növekszik / csökken
    bool buy = 0; /// adni vagy venni kell
    bool toresAlatt = 0; /// kisebb vaagy nagyobb legyen a törésnél?
    float tores = 0.0039f;

    bool operator==(const Parameterek& other) const {
        return (adasVeteliNapok==other.adasVeteliNapok &&
                m1==other.m1 &&
                m2==other.m2 &&
                m3==other.m3 &&
                ms==other.ms &&
                mi==other.mi &&
                buy==other.buy &&
                toresAlatt==other.toresAlatt// &&
                //tores==other.tores
                );
    }
};

struct Score{
    vector<float> evVegiek, teljes;
    vector<int> alkalmakEvente;
    int egyNapMaximum=0;
    float atlagosProfit=0;
    float atlagosNapiProfit=0;
    Parameterek param;

    float maxLoss = 2.0f, minProfit = 0.0f;

    void clrt(){
        evVegiek.clear(); evVegiek.resize(25);
        teljes.clear(); teljes.resize(6273);
        alkalmakEvente.clear(); alkalmakEvente.resize(25);
        egyNapMaximum=0;
        atlagosNapiProfit=0;
        atlagosProfit=0;
    }

    void chkProf(){
        for (int i=0; i<25; i++){
            if (i==0){
                maxLoss=evVegiek[0]/100.0f;
                minProfit=evVegiek[0]/100.0f;
            } else {
                maxLoss=min(evVegiek[i]/evVegiek[i-1],maxLoss);
                minProfit=max(evVegiek[i]/evVegiek[i-1],minProfit);
            }
        }
    }

    void print(){
        cout<<param.m1<<" "<<param.m2<<" "<<param.m3<<" "<<param.adasVeteliNapok<<" "<<param.buy<<" ";
        cout<<param.mi<<" "<<param.ms<<" "<<param.tores<<" "<<param.toresAlatt<<endl;
        cout<<egyNapMaximum<<" "<<atlagosProfit<<" "<<atlagosNapiProfit<<endl;
        for (int i=0; i<25; i++)
            cout<<alkalmakEvente[i]<<" ";
        cout<<endl;
        for (int i=0; i<25; i++)
            cout<<evVegiek[i]<<" ";
        cout<<endl;
        cout<<maxLoss<<" "<<minProfit<<endl;
    }

    void pt(){
        for(int i=0; i<teljes.size(); i++){
            cout<<teljes[i]<<" ";
        }
        cout<<endl;
    }


    void opp(ofstream& of){
        of<<param.adasVeteliNapok<<" "<<param.buy<<" ";
        of<<param.mi<<" "<<param.ms<<" "<<param.tores<<" "<<param.toresAlatt<<" ";
    }
    void op(ofstream& of){
        of<<param.m1<<" "<<param.m2<<" "<<param.m3<<" "<<param.adasVeteliNapok<<" "<<param.buy<<" ";
        of<<param.mi<<" "<<param.ms<<" "<<param.tores<<" "<<param.toresAlatt<<endl;
        of<<egyNapMaximum<<" "<<atlagosProfit<<" "<<atlagosNapiProfit<<endl;
        for (int i=0; i<25; i++)
            of<<alkalmakEvente[i]<<" ";
        of<<endl;
        for (int i=0; i<25; i++)
            of<<evVegiek[i]<<" ";
        of<<endl;
        of<<maxLoss<<" "<<minProfit<<endl;
    }
    void opt(ofstream& of){
        for(int i=0; i<teljes.size(); i++){
            of<<teljes[i]<<" ";
        }
        of<<endl;
    }
};

struct Par2{
    float ertek;
    float sorszam;

    bool operator<(const Par2& other) const {
        return ertek<other.ertek;
    }
};

void vizsgal(vector<Score>& scores, int parm, vector<float>& foSumok, int j){
    foSumok[j] = 0;
    vector<Par2> parok;
    for (int zzz=parm; zzz<scores[0].teljes.size()-10; zzz++){
        parok.clear(); parok.reserve(scores.size());
        for (int i=0; i<scores.size(); i++){
            Par2 par; par.sorszam=i;
            par.ertek = scores[i].teljes[zzz]/scores[i].teljes[zzz-parm];
            parok.push_back(par);
        }
        sort(parok.begin(),parok.end());
        reverse(parok.begin(), parok.end());
        float szum=0;
        for (int i=0; i<100; i++){
            szum+=scores[i].teljes[zzz+10]/scores[i].teljes[zzz+1];
        }
        foSumok[j]+=szum/100.0f;
    }
}

int main()
{
    string fileName = "0000_2000_AllSaveNeg.txt";
    ifstream iFileTest(fileName);
    string line; vector<string> lineSplit;
    int cnt=0;
    while (getline(iFileTest,line)) cnt++;
    cout<<cnt/6<<endl;

    ifstream ifile(fileName);
    vector<Score> scores; scores.reserve(cnt/6+1);
    clock_t t0 = clock();
    float maxProf = 0;
    while (!ifile.eof()){
        if (clock()>t0+5000){
            cout<<"Scores: "<<scores.size()<<endl;
            t0=clock();
        }
        Score score; score.clrt();

        ///cout<<"A "<<scores.size()<<endl;
        lineSplit.clear();
        getline(ifile,line);
        lineSplit = split(line," ");
        if (lineSplit.size()==0) break;
        ///cout<<lineSplit.size()<<endl;

        score.param.m1 = stoi(lineSplit[0]); score.param.m2 = stoi(lineSplit[1]); score.param.m3 = stoi(lineSplit[2]);
        score.param.adasVeteliNapok = stoi(lineSplit[3]); score.param.buy = stoi(lineSplit[4]); score.param.mi = stoi(lineSplit[5]);
        score.param.ms = stoi(lineSplit[6]); score.param.tores = stof(lineSplit[7]); score.param.toresAlatt = stoi(lineSplit[8]);
        ///cout<<lineSplit[0];
        ///cout<<"B "<<scores.size()<<endl;
        if (
            (score.param.adasVeteliNapok!=1 ||
            score.param.toresAlatt!=0) ||
            //score.param.mi!=2 ||
            score.param.tores<0.01f) {
            ///for (int i=0; i<5; i++) getline(ifile,line);
            ///continue;
        }

        lineSplit.clear();
        getline(ifile,line);
        lineSplit = split(line," ");
        score.egyNapMaximum = stoi(lineSplit[0]); score.atlagosProfit = stof(lineSplit[1]); score.atlagosNapiProfit = stof(lineSplit[2]);
        ///cout<<"C "<<scores.size()<<endl;

        lineSplit.clear();
        getline(ifile,line);
        lineSplit = split(line," ");
        for (int i=0; i<25; i++)
            score.alkalmakEvente[i] = stoi(lineSplit[i]);
        ///cout<<"D "<<scores.size()<<endl;

        lineSplit.clear();
        getline(ifile,line);
        lineSplit = split(line," ");
        for (int i=0; i<25; i++)
            score.evVegiek[i] = stof(lineSplit[i]);
        ///cout<<"E "<<scores.size()<<endl;

        lineSplit.clear();
        getline(ifile,line);
        lineSplit = split(line," ");
        score.maxLoss = stof(lineSplit[0]); score.minProfit = stof(lineSplit[1]);
        ///cout<<"F "<<scores.size()<<endl;

        lineSplit.clear();
        getline(ifile,line);
        lineSplit = split(line," ");
        for (int i=0; i<6273; i++)
            score.teljes[i] = stof(lineSplit[i]);
        ///cout<<"G "<<scores.size()<<endl;

        maxProf=max(maxProf,score.evVegiek[24]);

        scores.push_back(score);
    }

    bool secondFile = true;
    if (secondFile){
        ifstream ifile2("0000_2000_AllSavePoz.txt");
        while (!ifile2.eof()){
            if (clock()>t0+5000){
                cout<<"Scores: "<<scores.size()<<endl;
                t0=clock();
            }
            Score score; score.clrt();

            ///cout<<"A "<<scores.size()<<endl;
            lineSplit.clear();
            getline(ifile2,line);
            lineSplit = split(line," ");
            if (lineSplit.size()==0) break;
            ///cout<<lineSplit.size()<<endl;

            score.param.m1 = stoi(lineSplit[0]); score.param.m2 = stoi(lineSplit[1]); score.param.m3 = stoi(lineSplit[2]);
            score.param.adasVeteliNapok = stoi(lineSplit[3]); score.param.buy = stoi(lineSplit[4]); score.param.mi = stoi(lineSplit[5]);
            score.param.ms = stoi(lineSplit[6]); score.param.tores = stof(lineSplit[7]); score.param.toresAlatt = stoi(lineSplit[8]);
            ///cout<<lineSplit[0];
            ///cout<<"B "<<scores.size()<<endl;
            if (
                (score.param.adasVeteliNapok!=1 ||
                score.param.toresAlatt!=0) ||
                //score.param.mi!=2 ||
                score.param.tores<0.01f) {
                ///for (int i=0; i<5; i++) getline(ifile,line);
                ///continue;
            }

            lineSplit.clear();
            getline(ifile2,line);
            lineSplit = split(line," ");
            score.egyNapMaximum = stoi(lineSplit[0]); score.atlagosProfit = stof(lineSplit[1]); score.atlagosNapiProfit = stof(lineSplit[2]);
            ///cout<<"C "<<scores.size()<<endl;

            lineSplit.clear();
            getline(ifile2,line);
            lineSplit = split(line," ");
            for (int i=0; i<25; i++)
                score.alkalmakEvente[i] = stoi(lineSplit[i]);
            ///cout<<"D "<<scores.size()<<endl;

            lineSplit.clear();
            getline(ifile2,line);
            lineSplit = split(line," ");
            for (int i=0; i<25; i++)
                score.evVegiek[i] = stof(lineSplit[i]);
            ///cout<<"E "<<scores.size()<<endl;

            lineSplit.clear();
            getline(ifile2,line);
            lineSplit = split(line," ");
            score.maxLoss = stof(lineSplit[0]); score.minProfit = stof(lineSplit[1]);
            ///cout<<"F "<<scores.size()<<endl;

            lineSplit.clear();
            getline(ifile2,line);
            lineSplit = split(line," ");
            for (int i=0; i<6273; i++)
                score.teljes[i] = stof(lineSplit[i]);
            ///cout<<"G "<<scores.size()<<endl;

            maxProf=max(maxProf,score.evVegiek[24]);

            scores.push_back(score);
        }
    }
    cout<<"Final size: "<<scores.size()<<" "<<maxProf<<endl;

    ///int zzz=23;
    ///int parm = 100;
    ofstream ofileOK("maxLossNeg.txt");
    vector<Par2> parok;
    clock_t t3 = clock();
    for (int parm=10; parm<2000 && false; parm++){
        cout<<parm<<" "<<t3-clock()<<endl;
        t3=clock();
        float foSum = 0;
        for (int zzz=parm; zzz<scores[0].teljes.size()-10; zzz++){
            parok.clear(); parok.reserve(scores.size());
            for (int i=0; i<scores.size(); i++){
                ///if (i%10000 == 0) cout<<"I: "<<i<<" "<<scores.size()<<endl;
                Par2 par; par.sorszam=i;
                par.ertek = scores[i].teljes[zzz]/scores[i].teljes[zzz-parm];
                parok.push_back(par);
            }
            sort(parok.begin(),parok.end());
            reverse(parok.begin(), parok.end());
            ///ofstream ofileMaxLoss("maxLoss.txt");
            float szum=0;
            for (int i=0; i<100; i++){
                //cout<<"I2: "<<i<<" "<<parok.size()<<endl;
                ///cout<<scores[parok[i].sorszam].evVegiek[zzz+1]/scores[parok[i].sorszam].evVegiek[zzz]<<" ";
                szum+=scores[i].teljes[zzz+10]/scores[i].teljes[zzz+1];
                ///scores[parok[i].sorszam].opt(ofileMaxLoss);
            }
            foSum+=szum/100.0f;
            //ofileOK<<parm<<" "<<szum/100.0f<<endl;
        }
        ofileOK<<parm<<" "<<foSum/(scores[0].teljes.size()-10-parm)<<endl;
    }


    int thCnt = 8;
    vector<thread> szalak; szalak.resize(thCnt);
    vector<float> ertekek; ertekek.resize(thCnt);

    int startLim = 400, endLim = 600;
    clock_t t2 = clock();
    t3 = clock();
    ofstream ofile("nalassuk13.txt");
    for (size_t i=startLim; i<endLim;){
        int savedI = i;
        for (int j=0; j<thCnt; j++){
            szalak[j] = thread(vizsgal,ref(scores),i,ref(ertekek),j);
            i++;

            if (i>=endLim) break;
        }

        float ts = 0;
        for (int j=0; j<thCnt; j++){
            if (savedI+j>=endLim) continue;
            clock_t t1 = clock();
            if (szalak[j].joinable())
                szalak[j].join();

            if (clock()-t2>=20000){
                cout<<savedI+j<<", "<<clock()-t2<<" i "<<ts<<" - "<<(float)(clock()-t3)/(float)(savedI+j-0)<<endl;
                t2=clock();
            }

            ofile<<savedI+j<<" "<<ertekek[j]/(scores[0].teljes.size()-10-savedI-j)<<endl;
            //ofile<<savedI+j<<" "<<ertekek[j]<<" "<<(scores[0].teljes.size()-10-savedI+j)<<endl;


            continue;
            return 0;
        }

    }



    return 0;

    t2=clock();
    vector<Parameterek> parameterek;
                for (int j=0; j<2; j++){
                    for (int k=0; k<6;k++){
                        for (int l=0; l<4;l++){
                            for (int b=0;b<2; b++){
                                for (int u=0; u<2; u++){
                                    for (float f=-0.01f; f<=0.01f; f+=0.0002f){
                                        Parameterek params;
                                        ///params.m1=i1; params.m2=i2; params.m3=i3;
                                        params.adasVeteliNapok=j;
                                        params.tores=f;
                                        params.ms=k;
                                        params.mi=l;
                                        params.toresAlatt=u;
                                        params.buy=b;
                                        parameterek.push_back(params);
                                        cnt++;
                                        ///getScore(reszvenyek,params);
                                    }
                                }
                            }
                        }
                    }
                }

    vector<vector<float>> mennyireBecsuli(25, vector<float>(25));
    for(int i=0; i<25; i++){

        for (int j=0; j<25; j++){
            if (j<i) {cout<<"0 "; continue;}
            float szum=0, poz=0;
            for (int k=0; k<scores.size(); k++){
                szum++;
                if (i!=j) {if (scores[k].evVegiek[j]>scores[k].evVegiek[i]) poz++;}
                else {if (scores[k].evVegiek[j]>100) poz++;}
                //if (scores[k].evVegiek[i]<100 && scores[k].evVegiek[j]<scores[k].evVegiek[i]) poz++;
                //if (scores[k].evVegiek[i]>100 && scores[k].evVegiek[j]>scores[k].evVegiek[i]) poz++;
            }
            cout<<poz/szum<<" ";
        }
        cout<<endl;
    }
    cout<<"OK: "<<clock()-t2<<endl;
    int cntSum = 0;
    ofstream ofileParams("paramSum.txt");
    for (int i=0; i<parameterek.size(); i++){
        cnt = 0;

        vector<Score> selectedScores;
        for (int k=0; k<scores.size(); k++){
            if (scores[k].param.buy==parameterek[i].buy &&
                scores[k].param.mi==parameterek[i].mi &&
                scores[k].param.ms==parameterek[i].ms &&
                scores[k].param.adasVeteliNapok==parameterek[i].adasVeteliNapok &&
                scores[k].param.toresAlatt==parameterek[i].toresAlatt &&
                scores[k].param.tores>parameterek[i].tores-0.0001f &&
                scores[k].param.tores<parameterek[i].tores+0.0001f)
                {
                    selectedScores.push_back(scores[k]);
                    cnt++;
                }
        }

        if (cnt!=0){
            selectedScores[0].opp(ofileParams);
            cout<<cnt<<endl;
            float poz25 = 0, poz15 = 0, poz10 = 0, poz5 = 0;
            float sum25 = 0, sum15 = 0, sum10 = 0, sum5 = 0;
            float sumA[25], pozA[25];
            for (int k=0; k<selectedScores.size(); k++){
                if (selectedScores[k].evVegiek[0]>100) {poz25++; pozA[0]++;}
                sum25++; sumA[0]++;
                for (int j=1; j<10; j++){
                    if (selectedScores[k].evVegiek[j-1]<selectedScores[k].evVegiek[j]) {poz25++;pozA[j]++;}
                    sum25++; sumA[j]++;
                }
                for (int j=10; j<15; j++){
                    if (selectedScores[k].evVegiek[j-1]<selectedScores[k].evVegiek[j]) {poz25++;poz15++;pozA[j]++;}
                    sum25++; sum15++; sumA[j]++;
                }
                for (int j=15; j<20; j++){
                    if (selectedScores[k].evVegiek[j-1]<selectedScores[k].evVegiek[j]) {poz25++;poz15++;poz10++;pozA[j]++;}
                    sum25++; sum15++; sum10++; sumA[j]++;
                }
                for (int j=20; j<25; j++){
                    if (selectedScores[k].evVegiek[j-1]<selectedScores[k].evVegiek[j]) {poz25++;poz15++;poz10++;poz5++;pozA[j]++;}
                    sum25++; sum15++; sum10++; sum5++; sumA[j]++;
                }
            }
            for (int k=0; k<25; k++){
                ofileParams<<(pozA[k]/sumA[k])<<" ";
            }
            ofileParams<<(poz25/sum25)<<" "<<(poz15/sum15)<<" "<<(poz10/sum10)<<" "<<(poz5/sum5)<<" ";
            ofileParams<<(sum25)<<" "<<(sum15)<<" "<<(sum10)<<" "<<(sum5)<<" "<<endl;
        }
        cntSum+=cnt;
    }
    cout<<"OK: "<<clock()-t2<<" "<<cntSum<<endl;

    //Sleep(100000);

    return 0;
}
