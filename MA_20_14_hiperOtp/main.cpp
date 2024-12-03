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

/// Függvény, amely beolvassa a "stocks.txt" fájlból a részvény neveket egy set-be.
std::unordered_set<std::string> readStockNames(const std::string& filename) {
    std::unordered_set<std::string> stockNames;
    std::ifstream file(filename);
    if (!file.is_open()) {
        std::cerr << "Hiba a fájl megnyitásakor: " << filename << std::endl;
        return stockNames;
    }

    std::string line;
    while (std::getline(file, line)) {
        if (!line.empty()) {
            for (char& c : line) {
                c = std::tolower(c);
            }
            stockNames.insert(line); // Részvény neveket egy set-be mentjük.
        }
    }

    return stockNames;
}

/// Függvény, amely bejárja a megadott mappát, és kigyûjti a megfelelõ fájlok elérési útvonalát.
void findStockFiles(const std::unordered_set<std::string>& stockNames, const std::string& directory, std::vector<std::string>& foundFiles) {
    std::string searchPath = directory + "\\*.us.txt";
    WIN32_FIND_DATA findFileData;
    HANDLE hFind = FindFirstFile(searchPath.c_str(), &findFileData);

    if (hFind == INVALID_HANDLE_VALUE) {
        std::cerr << "Nem található fájl ebben a mappában: " << directory << std::endl;
        return;
    }

    do {
        std::string filename = findFileData.cFileName;

        // Levágjuk a ".us.txt" részt a fájl nevébõl.
        std::string stockName = filename.substr(0, filename.find(".us.txt"));

        // Ellenõrizzük, hogy a részvény neve szerepel-e a listában.
        if (stockNames.find(stockName) != stockNames.end()) {
            foundFiles.push_back(directory + "\\" + filename);
        }

    } while (FindNextFile(hFind, &findFileData) != 0);

    FindClose(hFind);
}

/// Függvény, ami összegyűjti a részvények adatait tároló fájlok elérési útvonalát
vector<string> reszvenyekEleresiUtja(string fname="stocks.txt", string folder="data"){
    std::string stockListFile = fname; // A részvény nevek fájlja
    std::unordered_set<std::string> stockNames = readStockNames(stockListFile);

    std::vector<std::string> foundFiles;

    if (stockNames.empty()) {
        std::cerr << "Nincs elérhetõ részvény név." << std::endl;
        return foundFiles;
    }


    std::vector<std::string> directories = {
        "\\daily\\us\\nasdaq stocks\\1",
        "\\daily\\us\\nasdaq stocks\\2",
        "\\daily\\us\\nasdaq stocks\\3",
        "\\daily\\us\\nyse stocks\\1",
        "\\daily\\us\\nyse stocks\\2"
    };

    // Végigmegyünk az összes mappán és megkeressük a fájlokat.
    for (const auto& dir : directories) {
        stringstream ss; ss<<folder<<dir;
        findStockFiles(stockNames, ss.str(), foundFiles);
    }

    // Kiírjuk az összegyûjtött fájlok elérési útvonalait.
    ///std::cout << "Talált fájlok:\n";
    for (const auto& file : foundFiles) {
        std::cout << file << std::endl;
    }
    return foundFiles;
}

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

struct Datum{
    int ev, honap, nap;

    Datum& operator=(const Datum& other) {
        if (this != &other) { // Önhozzárendelés ellenőrzése
            this->ev = other.ev;
            this->honap = other.honap;
            this->nap = other.nap;
        }
        return *this; // Láncolhatóvá tesszük
    }

    Datum(int e, int h, int n){ev=e; honap=h; nap=n;}
    Datum(){ev=0; honap=0; nap=0;}

    int toInt(){return ev*10000+honap*100+nap;}
    Datum fromInt(int d){
        Datum datum; datum.ev=d/10000;
        datum.honap=(d%10000)/100;
        datum.nap=d%100;
        return datum;
    }

    void setInt(int d){ev=d/10000;
        honap=(d%10000)/100;
        nap=d%100;
    }

    string toString(){
        stringstream ss; ss<<ev<<" "<<honap<<" "<<nap;
        return ss.str();
    }

    string toStringDate(){
        stringstream ss; ss<<ev<<"."<<honap<<"."<<nap<<".";
        return ss.str();
    }

    bool operator==(const Datum& other) const {
        return (ev==other.ev && honap==other.honap && nap==other.nap);
    }

    bool operator<(const Datum& other) const {
        if (ev<other.ev) return true;
        if (ev==other.ev && honap<other.honap) return true;
        if (ev==other.ev && honap==other.honap && nap<other.nap) return true;
        return false;
    }
};

struct MozgoAtlag{
    int hossz;
    vector<float> atlag;
    vector<Datum> datum;
    vector<float> zaras;
    vector<float> nyitas;
};

struct Nap{
    float nyitas, minimum, maximum, zaras;
    Datum datum;
};

struct Reszveny{
    string nev;
    vector<Nap> napok;
    vector<MozgoAtlag> mozgoatlag;
};

struct ReszvenyGPU{
    vector< vector<float> > mozgoatlagokAtlag;
    vector<float> mozgoatlagokZaras;
    vector<int> mozgoatlagokDatum;
    string nev;
    int N;
};

/// Betölti a fájlokból a részvények adatait
vector<Reszveny> reszvenyekBetoltese(vector<string> fajlnevek){
    vector<Reszveny> reszvenyek;
    /// most az adatokat kell beolvasni az adott fájlokból
    for (size_t i=0; i<fajlnevek.size(); i++){
        ifstream ifileStockData(fajlnevek[i]);
        Reszveny reszveny;
        string reszvenyNeve,fejlec,toSplit;
        ifileStockData>>fejlec;
        vector<string> elemek;
        reszveny.mozgoatlag.resize(200);
        while (!ifileStockData.eof()){
            ifileStockData>>toSplit;
            elemek = split(toSplit,",");
            stringstream ss(elemek[2]); /// 19871230 -> 1987 12 30
            int datum=0; ss>>datum;
            Nap nap; nap.datum.ev=datum/10000; nap.datum.nap=datum%100;
            nap.datum.honap=(datum%10000)/100;
            nap.nyitas=stof(elemek[4]);
            nap.maximum=stof(elemek[5]);
            nap.minimum=stof(elemek[6]);
            nap.zaras=stof(elemek[7]);
            reszveny.napok.push_back(nap);

            for (float n=1.0f; n<=200; n++){
                if (reszveny.napok.size()<n) continue;
                if (reszveny.napok.size()==n) {
                    /*
                    MozgoAtlag ujAtlag; ujAtlag.datum=nap.datum;
                    for (int k=0; k<200;k++) ujAtlag.m200+=reszveny.napok[k].zaras;
                    ujAtlag.m200/=200.0f;
                    for (int k=100; k<200;k++) ujAtlag.m100+=reszveny.napok[k].zaras;
                    ujAtlag.m100/=100.0f;
                    for (int k=150; k<200;k++) ujAtlag.m50+=reszveny.napok[k].zaras;
                    ujAtlag.m50/=50.0f;
                    ujAtlag.zaras=reszveny.napok[199].zaras;
                    reszveny.mozgoatlag.push_back(ujAtlag);
                    */
                    continue;
                }
                reszveny.mozgoatlag[n-1].hossz=n;
                reszveny.mozgoatlag[n-1].datum.push_back(nap.datum);
                reszveny.mozgoatlag[n-1].zaras.push_back(nap.zaras);
                reszveny.mozgoatlag[n-1].nyitas.push_back(nap.nyitas);
                float atlag = 0;
                for (int k=reszveny.napok.size()-n; k<reszveny.napok.size();k++)
                    atlag+=reszveny.napok[k].zaras;
                atlag/=n;
                reszveny.mozgoatlag[n-1].atlag.push_back(atlag);
                /*
                MozgoAtlag ujAtlag = reszveny.mozgoatlag.back();
                int index = reszveny.napok.size()-201;
                ujAtlag.datum = nap.datum;
                ujAtlag.m200+=nap.zaras/200.0f;
                ujAtlag.m200-=reszveny.napok[index].zaras/200.0f;
                ujAtlag.m100+=nap.zaras/100.0f;
                ujAtlag.m100-=reszveny.napok[index+100].zaras/100.0f;
                ujAtlag.m50+=nap.zaras/50.0f;
                ujAtlag.m50-=reszveny.napok[index+150].zaras/50.0f;
                ujAtlag.zaras=reszveny.napok[index+200].zaras;
                reszveny.mozgoatlag.push_back(ujAtlag);
                */
            }
        }
        elemek = split(elemek[0],".");
        reszveny.nev=elemek[0];
        cout<<reszveny.nev<<endl;
        reszvenyek.push_back(reszveny);
    }
    return reszvenyek;
}

void reszvenyBetoltese(Reszveny& reszveny, string fajlnev){
    reszveny.mozgoatlag.clear(); reszveny.napok.clear(); reszveny.nev="";
    ifstream ifileStockData(fajlnev);
    string reszvenyNeve,fejlec,toSplit;
    //ifileStockData>>fejlec;
    getline(ifileStockData,toSplit);
    vector<string> lines; lines.reserve(10000);
    ///cout<<"OK"<<endl;
    int MOZGO_CNT = 50;
    while (!ifileStockData.eof()){
        getline(ifileStockData,toSplit);
        if (toSplit.size()>2)
            lines.push_back(toSplit);
    }
    ///cout<<"OK2"<<endl;
    vector<string> elemek;
    reszveny.mozgoatlag.resize(MOZGO_CNT);
    vector<list<MozgoAtlag>> mozgAtlgk(MOZGO_CNT);
    for (string str: lines){
        ///cout<<"OK3"<<endl;
        ///cout<<str<<endl;
        //ifileStockData>>toSplit;
        toSplit=str;
        elemek = split(toSplit,",");
        stringstream ss(elemek[2]); /// 19871230 -> 1987 12 30
        int datum=0; ss>>datum;
        Nap nap; nap.datum.ev=datum/10000; nap.datum.nap=datum%100;
        nap.datum.honap=(datum%10000)/100;
        nap.nyitas=stof(elemek[4]);
        nap.maximum=stof(elemek[5]);
        nap.minimum=stof(elemek[6]);
        nap.zaras=stof(elemek[7]);
        reszveny.napok.push_back(nap);
        ///if (nap.datum.ev<2000) continue;

        for (float n=1.0f; n<=MOZGO_CNT; n++){
            if (reszveny.napok.size()<n) continue;
            if (reszveny.napok.size()==n) {
                continue;
            }
            reszveny.mozgoatlag[n-1].hossz=n;
            reszveny.mozgoatlag[n-1].datum.push_back(nap.datum);
            reszveny.mozgoatlag[n-1].zaras.push_back(nap.zaras);
            reszveny.mozgoatlag[n-1].nyitas.push_back(nap.nyitas);
            float atlag = 0;
            for (int k=reszveny.napok.size()-n; k<reszveny.napok.size();k++)
                atlag+=reszveny.napok[k].zaras;
            atlag/=n;
            reszveny.mozgoatlag[n-1].atlag.push_back(atlag);
        }
    }
    elemek = split(elemek[0],".");
    reszveny.nev=elemek[0];
    ///cout<<reszveny.nev<<endl;
}

void valami(Reszveny& reszveny, string fajlnev){
    reszveny.mozgoatlag.clear(); reszveny.napok.clear(); reszveny.nev="";
    reszveny.mozgoatlag.resize(200);
    for (int i=0; i<1000;i++){
        int r = rand();
        Nap nap; nap.datum.ev=r%2; nap.datum.honap=r%4; nap.datum.nap=r%8;
        nap.maximum = r%3;
        nap.minimum = r%5;
        nap.nyitas = r%6;
        nap.zaras = r%7;
        reszveny.napok.push_back(nap);
        if (reszveny.napok.size()>200){
            for (float n=1.0f; n<=200; n++){
                if (reszveny.napok.size()<n) continue;
                if (reszveny.napok.size()==n) {
                    continue;
                }
                reszveny.mozgoatlag[n-1].hossz=n;
                reszveny.mozgoatlag[n-1].datum.push_back(nap.datum);
                reszveny.mozgoatlag[n-1].zaras.push_back(nap.zaras);
                reszveny.mozgoatlag[n-1].nyitas.push_back(nap.nyitas);
                float atlag = 0;
                for (int k=reszveny.napok.size()-n; k<reszveny.napok.size();k++)
                    atlag+=reszveny.napok[k].zaras;
                atlag/=n;
                reszveny.mozgoatlag[n-1].atlag.push_back(atlag);
            }
        }
    }
}

vector<Reszveny> reszvenyekParhuzamosBetoltese(vector<string> fajlnevek){
    vector<Reszveny> osszesReszveny; osszesReszveny.reserve(2000);
    int thCnt =4;
    vector<thread> szalak; szalak.resize(thCnt);
    vector<Reszveny> reszvenyek; reszvenyek.resize(thCnt);

    cout<<"CNT: "<<fajlnevek.size()<<endl;
    clock_t t1 = clock();
    clock_t t2 = clock();
    clock_t t3 = clock();
    for (size_t i=0; i<fajlnevek.size();){
        int savedI = i;
        t1=clock();
        for (int j=0; j<thCnt; j++){
            szalak[j] = thread(reszvenyBetoltese,ref(reszvenyek[j]),fajlnevek[i]);
            i++;
            if (i>=fajlnevek.size()) break;
        }
        //cout<<clock()-t1<<endl;

        float ts = 0;
        for (int j=0; j<thCnt; j++){
            if (savedI+j>=fajlnevek.size()) continue;
            clock_t t1 = clock();
            //if (szalak[j].joinable())
                szalak[j].join();
                //thread.
            if (clock()-t2>=5000){
                cout<<savedI+j<<" "<<clock()-t2<<" i "<<ts<<" - "<<(float)(clock()-t3)/(float)(savedI+j)<<endl;
                t2=clock();
            }
            osszesReszveny.push_back(reszvenyek[j]);
        }
        //cout<<clock()-t1<<endl;
    }
    return osszesReszveny;
}


char xedikKarakter(int X, string abc = "abcdefghijklmnopqrstuvwxyz"){
    int i = 0;
    for(char& c : abc) {
        i++;
        if (i==X) return c;
    }
    return '?';
}

int karakterSorszama(char X){
    string abc = "abcdefghijklmnopqrstuvwxyz";
    for(int i=0; i<20; i++) {
        if (abc[i]==X) return i;
    }
    return -1;
}

struct Par{
    float ertek;
    char karakter;

    bool operator<(const Par& other) const {
        return ertek<other.ertek;
    }
};

/*
set<float> convertToSet(vector<float> v)
{
    set<float> s;
    for (float x : v) {
        s.insert(x);
    }
    return s;
}
*/

template <typename T>
std::set<T> convertToSet(const std::vector<T>& v) {
    std::set<T> s;
    for (const T& x : v) {
        s.insert(x);
    }
    return s;
}


struct Pelda {
    string stockName;
    Datum datum;
    int reszvenyIdx, mozgoatlagIdx;

    bool operator<(const Pelda& other) const {
        if (datum==other.datum)
            return stockName<other.stockName;
        return datum<other.datum;
    }
};

struct PeldaGPU{
    int reszvenyIdx, mozgoatlagIdx, datum;

    bool operator<(const PeldaGPU& other) const {
        if (mozgoatlagIdx==other.mozgoatlagIdx)
            return reszvenyIdx<other.reszvenyIdx;
        return mozgoatlagIdx<other.mozgoatlagIdx;
    }
};

struct Eset{
    string charChain= "";
    float osszesEset = 0;
    float pozitivEset = 0;

    float szum = 0, prod = 1;

    vector<Pelda> peldak;

    bool operator<(const Eset& other) const {
        return charChain<other.charChain;
    }
    bool operator==(const Eset& other) const {
        return charChain==other.charChain;
    }

    // Kényelmes kiíráshoz egy barát operátor
    friend std::ostream& operator<<(std::ostream& os, const Eset& obj) {
        //return os << obj.charChain << ": " << obj.osszesEset << ", " << obj.pozitivEset << ", " << obj.szum << ", " << obj.prod;

        return os << obj.charChain << " "
        << obj.osszesEset<< " " << obj.pozitivEset << " "
        << obj.szum << " " << obj.prod << " "
        << obj.pozitivEset/obj.osszesEset*100.0f << " "
        << obj.szum/obj.osszesEset<<" ";
    }
};

struct Par2{
    float ertek;
    float sorszam;

    bool operator<(const Par2& other) const {
        return ertek<other.ertek;
    }
};

struct Ertek {
    int osszes=0, pozitiv=0;
    float szum=0, prod=1;
};

Datum xhonapra(Datum kezdo, int honapokSZama){
    for (int i=0; i<honapokSZama; i++){
        kezdo.honap++;
        if(kezdo.honap>12) {kezdo.ev++; kezdo.honap=1;}
    }
    return kezdo;
}

struct Tranzakcio{
    Datum datum;
    string reszveny;
    float ertek;

    bool operator<(const Tranzakcio& other) const {
        return datum<other.datum;
    }
};

struct Parameterek{
    int adasVeteliNapok = 0; /// (0-5)
    int m1 = 1, m2=2, m3=3; /// a három vizsgált mozgóátlag (1-50, különözőek)
    int ms = 0; /// mozgóátlagok növekvő sorrendje (0-5)
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
    vector<Pelda> osszesPelda;
    int egyNapMaximum=0;
    float atlagosProfit=0;
    float atlagosNapiProfit=0;
    Parameterek param;

    float maxLoss = 2.0f, minProfit = 0.0f;

    void clrt(){
        evVegiek.clear(); evVegiek.resize(25);
        teljes.clear(); osszesPelda.clear();
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

bool chkSorrend(const float& a, const float& b, const float& c){
    return (a<=b && b<=c);
}

int getScore2(vector<ReszvenyGPU>& reszvenyekGpu, Parameterek& params, vector<Score>& allScores, vector<Datum>& osszesDatum, int ert){
    clock_t t1 = clock();
    int N = reszvenyekGpu[0].N;
    int m1 = params.m1, m2 = params.m2, m3 = params.m3;
    //params.
    vector<vector<PeldaGPU>> peldak(4800); /// mi*ms*toresElojel 4*6*2=48 * 100 törés
    PeldaGPU p;
    if (ert==-1)cout<<"STEP 1 "<<clock()-t1<<endl;
    t1 = clock();
    for (int k=0;k<reszvenyekGpu.size();k++){
        ///cout<<"K: "<<k<<endl;
        for (int i=2; i<N-2; i++){
            ///cout<<"i1: "<<i<<endl;
            float ma11 = reszvenyekGpu[k].mozgoatlagokAtlag[m1][i-2];
            float ma21 = reszvenyekGpu[k].mozgoatlagokAtlag[m1][i-1];
            float ma31 = reszvenyekGpu[k].mozgoatlagokAtlag[m1][i];
            if (ma11==-1 || ma21==-1 || ma31==-1) continue;

            float ma12 = reszvenyekGpu[k].mozgoatlagokAtlag[m2][i-2];
            float ma22 = reszvenyekGpu[k].mozgoatlagokAtlag[m2][i-1];
            float ma32 = reszvenyekGpu[k].mozgoatlagokAtlag[m2][i];
            float ma42 = reszvenyekGpu[k].mozgoatlagokAtlag[m2][i+1];
            float ma52 = reszvenyekGpu[k].mozgoatlagokAtlag[m2][i+2];
            if (ma12==-1 || ma22==-1 || ma32==-1 || ma42==-1 || ma52==-1) continue;

            float ma13 = reszvenyekGpu[k].mozgoatlagokAtlag[m3][i-2];
            float ma23 = reszvenyekGpu[k].mozgoatlagokAtlag[m3][i-1];
            float ma33 = reszvenyekGpu[k].mozgoatlagokAtlag[m3][i];
            if (ma13==-1 || ma23==-1 || ma33==-1) continue;

            ///cout<<"i2: "<<i<<endl;

            float b1 = ma22-ma12, b2 = ma32-ma22;
            int elojel = 0; /// pozitív
            if (b2<0 && b1<0 && b1<b2) elojel=1; /// negatív
            else if(b2>0 && b1>0 && b1>b2) elojel=0;
            else continue;

            float tores = (b1-b2)/ma22;

            int sorrend = 0;
            if (chkSorrend(ma33,ma32,ma31)) sorrend = 0;
            else if (chkSorrend(ma33,ma31,ma32)) sorrend = 1;
            else if (chkSorrend(ma32,ma31,ma33)) sorrend = 2;
            else if (chkSorrend(ma32,ma33,ma31)) sorrend = 3;
            else if (chkSorrend(ma31,ma33,ma32)) sorrend = 4;
            else if (chkSorrend(ma31,ma32,ma33)) sorrend = 5;

            int tendencia = 0;
            if (ma33-ma23 <0) tendencia+=1;
            if (ma31-ma21 <0) tendencia+=2;

            int esetSzam = elojel*24 + tendencia*6 + sorrend;
            ///cout<<"i3: "<<i<<endl;
            p.datum=reszvenyekGpu[k].mozgoatlagokDatum[i];
            p.mozgoatlagIdx=i; p.reszvenyIdx=k;
            float itn;
            float su = modf(tores/0.0002f,&itn);
            if (su<0) itn-=1;
            if (itn+50>=100) itn=49;
            if (itn+50<0) itn = -50;
            ///cout<<"i4: "<<i<<" "<<itn<<endl;
            if (tores>0.00379f && tendencia==0 && sorrend==0){/// && reszvenyekGpu[k].nev=="AMD"){
            //if (reszvenyekGpu[k].nev=="AMD"){
                ///cout<<reszvenyekGpu[k].mozgoatlagokDatum[i]<<" "<<esetSzam<<" "<<tores<<" "<<b2<<" "<<b1<<" "<<b1-b2<<" "<<ma22<<" "<<reszvenyekGpu[k].nev<<" "<<itn<<endl;
            }
            if (itn<0){
                for (int u=itn+50; 0<=u && u<50; u++){
                    ///cout<<u<<" "<<esetSzam+u*48<<endl;
                    ///if (esetSzam+u*48==3312) cout<<"BAJ"<<endl;
                    peldak[esetSzam+u*48].push_back(p);
                }
            } else {
                for (int u=itn+50; 100>u && u>=50; u--){
                    ///if (esetSzam+u*48==3312) cout<<"BAJ "<<tendencia<<" "<<sorrend<<" "<<tores<<endl;
                    ///if (tores>0.00379f && tendencia==0 && sorrend==0 && esetSzam+u*48==3312) cout<<"OK "<<peldak[esetSzam+u*48].size()<<" "<<esetSzam+u*48<<endl;

                    peldak[esetSzam+u*48].push_back(p);
                }
            }
            ///cout<<"i5: "<<i<<endl;
        }
    }

    if (ert==-1)cout<<"STEP 2 "<<clock()-t1<<endl;
    t1 = clock();
    vector<Parameterek> paramok(4800);
    Parameterek pa;
    for (int i=0; i<6; i++){
        for (int j=0; j<4; j++){
            for (int k=0; k<2; k++){
                int z = 0;
                for (float u=-0.01f; u<0.01f;u+=0.0002f){///0.0039f
                    pa.m1=m1; pa.m2=m2; pa.m3=m3;
                    pa.toresAlatt=(k==1);
                    pa.tores=u;
                    pa.ms=i;
                    pa.mi=j;
                    int esetSzam = i+j*6+k*24+z*48;
                    if (z<100)
                        paramok[esetSzam]=pa;
                    z++;
                }
            }
        }
    }

    if (ert==-1)cout<<"STEP 3 "<<clock()-t1<<endl;
    t1 = clock();

    allScores.resize(4800*4);
    for (int zz=0;zz<4800*4;zz++)
        allScores[zz].clrt();
    if (ert==-1)cout<<"STEP 4 "<<clock()-t1<<endl;
    t1 = clock();
    for (int l=0; l<4800; l++){
            /*
        if (paramok[l].ms==params.ms && paramok[l].tores>0.0037f && paramok[l].tores<0.0039f && paramok[l].mi==params.mi){

        }
        else {
            //if (paramok[l].toresAlatt==false)
              //  cout<<paramok[l].adasVeteliNapok<<paramok[l].buy<<paramok[l].mi<<paramok[l].ms<<paramok[l].tores<<endl;
                //cout<<params.adasVeteliNapok<<params.buy<<params.mi<<params.ms<<params.tores<<endl;
            //
            continue;
        }
        */
        ///cout<<"STEP 4.1 "<<l<<endl;
        sort(peldak[l].begin(),peldak[l].end());
        //list<PeldaGPU> osszesPeldaList;
        //osszesPeldaList.insert(osszesPeldaList.end(),peldak[i].begin(),peldak[i].end());
        //cout<<peldak[i].size()<<" "<<osszesPeldaList.size()<<endl;

        vector<Score> scores(4);
        for (int zz=0;zz<4;zz++)
            scores[zz].clrt();
        if (peldak[l].size()==0) continue;

        //cout<<l<<" "<<peldak[l].size()<<" "<<peldak[l].size()<<endl;

        vector<vector<float>> napiErtek(4);
        for (int k=0; k<4; k++)
            napiErtek[k].resize(osszesDatum.size(),100.0f);
        //napiErtek.resize(osszesDatum.size(),100.0f);
        vector<float> tempSzum(4,0), tempCnt(4,0);
        vector<vector<float>> eviErtekek(4);
        vector<vector<vector<float>>> eviNapok(4);
        ///cout<<"STEP 4.2"<<endl;
        float peldasNapCnt=0;
        int tobeUsed = 0;
        int ev = 2000;
        for (int i=0; i<N; i++){
            ///cout<<"N1: "<<i<<endl;
            for (int zz=0;zz<4; zz++){
                if (i+1<napiErtek[zz].size()){
                    napiErtek[zz][i+1]=napiErtek[zz][i];
                    ///cout<<zz<<" "<<i<<endl;
                }
            }
            if (osszesDatum[i].ev>ev){
                ev=osszesDatum[i].ev;
                ///napiErtek[i+0]=100;
                ///napiErtek[i+1]=100;
                ///cout<<"HALI"<<endl;
                for (int zz=0;zz<4;zz++){
                    scores[zz].evVegiek[ev-2001]=eviErtekek[zz].back();
                    ///cout<<scores[zz].evVegiek[ev-2001]<<" ";
                    eviNapok[zz].push_back(eviErtekek[zz]);
                    eviErtekek[zz].clear();
                }
                ///cout<<endl;
            }
            ///cout<<"N2: "<<i<<endl;
            vector<vector<float>> ertekek(4);
            ///if (datumok.find(osszesDatum[i])==datumok.end()){eviErtekek.push_back(napiErtek[i]); continue;}
            //for (int zz=0;zz<4; zz++)
              //  eviErtekek[zz].push_back(napiErtek[zz][i]);
            ///for (int j=0; j<osszesPelda.size(); j++){
            bool vanPelda = false;
            if (tobeUsed<peldak[l].size()){
                ///cout<<tobeUsed<<" "<<peldak[l].size()<<" "<<i<<" "<<endl;
                while(peldak[l][tobeUsed].datum==osszesDatum[i].toInt()){

                    ///cout<<reszvenyekGpu[peldak[l][tobeUsed].reszvenyIdx].nev<<endl;
                    if (reszvenyekGpu[peldak[l][tobeUsed].reszvenyIdx].nev=="AMD"){
                       /// cout<<osszesDatum[i].toString()<<endl;
                    }
                    ///cout<<tobeUsed<<" "<<peldak[l].size()<<" "<<i<<" ";
                    vanPelda=true;
                    int stockIdx = peldak[l][tobeUsed].reszvenyIdx;
                    int mozgoAtlagIdx = peldak[l][tobeUsed].mozgoatlagIdx;
                    ///cout<<mozgoAtlagIdx<<" "<<stockIdx<<endl;
                    if (mozgoAtlagIdx<2 || mozgoAtlagIdx>N-3) cout<<"ZZZZZZZ"<<endl;
                    float aznapiZaras = reszvenyekGpu[stockIdx].mozgoatlagokZaras[mozgoAtlagIdx];
                    float masnapiZaras = reszvenyekGpu[stockIdx].mozgoatlagokZaras[mozgoAtlagIdx+1];
                    float harmadnapiZaras = reszvenyekGpu[stockIdx].mozgoatlagokZaras[mozgoAtlagIdx+2];

                    float e0 = masnapiZaras/aznapiZaras-1.0f;
                    float e1 = -(masnapiZaras/aznapiZaras-1.0f);
                    float e2 = harmadnapiZaras/masnapiZaras-1.0f;
                    float e3 = -(harmadnapiZaras/masnapiZaras-1.0f);

                    if (e0>0.8f || e1>0.8f || e2>0.8f || e3>0.8f){
                        tobeUsed++;
                        if (tobeUsed>=peldak[l].size()){break;}
                        continue;
                    }

                    ertekek[0].push_back(masnapiZaras/aznapiZaras-1.0f);
                    ertekek[1].push_back(-(masnapiZaras/aznapiZaras-1.0f));
                    ertekek[2].push_back(harmadnapiZaras/masnapiZaras-1.0f);
                    ertekek[3].push_back(-(harmadnapiZaras/masnapiZaras-1.0f));

                    if (ertekek[0].back()>2) cout<<"HUPSZ0 "<<masnapiZaras<<" "<<aznapiZaras<<" "<<reszvenyekGpu[stockIdx].nev<<" "<<reszvenyekGpu[stockIdx].mozgoatlagokDatum[mozgoAtlagIdx]<<endl;
                    if (ertekek[1].back()>2) cout<<"HUPSZ1 "<<masnapiZaras<<" "<<aznapiZaras<<" "<<reszvenyekGpu[stockIdx].nev<<" "<<reszvenyekGpu[stockIdx].mozgoatlagokDatum[mozgoAtlagIdx]<<endl;
                    if (ertekek[2].back()>2) cout<<"HUPSZ2 "<<harmadnapiZaras<<" "<<masnapiZaras<<" "<<reszvenyekGpu[stockIdx].nev<<" "<<reszvenyekGpu[stockIdx].mozgoatlagokDatum[mozgoAtlagIdx]<<endl;
                    if (ertekek[3].back()>2) cout<<"HUPSZ3 "<<harmadnapiZaras<<" "<<masnapiZaras<<" "<<reszvenyekGpu[stockIdx].nev<<" "<<reszvenyekGpu[stockIdx].mozgoatlagokDatum[mozgoAtlagIdx]<<endl;

                    tobeUsed++;
                    if (tobeUsed>=peldak[l].size()){break;}
                }
            }
            ///cout<<"N3: "<<i<<endl;
            if (vanPelda) peldasNapCnt++;
            int oszto = ertekek[0].size();
            float foszto = oszto;
            for (int zz=0;zz<4; zz++){
                scores[zz].egyNapMaximum=max(oszto,scores[zz].egyNapMaximum);
                scores[zz].alkalmakEvente[ev-2000]+=oszto;
                ///cout<<score.egyNapMaximum<<endl;

                for (int j=0; j<oszto; j++){
                    float tempSum = 0;
                    if (i+1<napiErtek[zz].size()){
                        ///if (ertekek[zz][j]/foszto > 1) cout<<"AJJAJ"<<endl;
                        napiErtek[zz][i+1]+=napiErtek[zz][i]*ertekek[zz][j]/foszto;
                        tempSzum[zz]+=ertekek[zz][j];
                        tempSum+=ertekek[zz][j];;
                        tempCnt[zz]++;
                    }
                    scores[zz].atlagosNapiProfit+=tempSum/foszto;
                }
                eviErtekek[zz].push_back(napiErtek[zz][i]);
            }
            ///cout<<"N4: "<<i<<endl;
        }
        ///cout<<"STEP 4.3"<<endl;
        for (int zz=0;zz<4; zz++){
            scores[zz].atlagosNapiProfit/=max(1.0f,peldasNapCnt);
            clock_t endtime=clock();
            scores[zz].evVegiek[24]=eviErtekek[zz].back();
            ///cout<<peldak[l].size()<<" "<<scores[zz].evVegiek[24]<<endl;
            eviNapok[zz].push_back(eviErtekek[zz]);
            ///cout<<"CALMA "<<ert<<endl;
            scores[zz].atlagosProfit=tempSzum[zz]/max(1.0f,tempCnt[zz]);
            scores[zz].teljes=napiErtek[zz];
            scores[zz].param=paramok[l];
        }
        scores[0].param.buy = true;
        scores[0].param.adasVeteliNapok = 0;
        scores[1].param.buy = false;
        scores[1].param.adasVeteliNapok = 0;
        scores[2].param.buy = true;
        scores[2].param.adasVeteliNapok = 1;
        scores[3].param.buy = false;
        scores[3].param.adasVeteliNapok = 1;
        for (int zz=0;zz<4; zz++){
            ///cout<<scores[zz].evVegiek[24]<<endl;
            scores[zz].chkProf();
            allScores[l*4+zz]=scores[zz];
        }
    }

    if (ert==-1)cout<<"STEP 5 "<<clock()-t1<<endl;
    t1 = clock();
    return 0;
}

int getScore(vector<Reszveny>& reszvenyek, Parameterek& params, Score& score, vector<Datum>& osszesDatum, int ert){
    score.clrt();
    clock_t stime = clock();
    int MOZGO_ATLAG_MIN_MERET = 50;

    bool debug = false;

    vector<Datum> osszesEset; osszesEset.reserve(100000);
    vector<Pelda> osszesPelda; osszesPelda.reserve(100000);
    vector<PeldaGPU> osszesPeldaG;
    ///cout<<"ALMA "<<ert<<endl;
    ///return 0;
    Pelda pelda;
    int itrCnt = 0, itrCnt2 = 0, itrCnt3 = 0;
    for (int i=0;i<reszvenyek.size(); i++){
        ///cout<<"R"<<i<<endl;
        for (int k=2; k<reszvenyek[i].mozgoatlag[params.m2].atlag.size(); k++){
            itrCnt++;
            /// megvizsgálja hogy az adott nap megfelelő-e
            int ev = reszvenyek[i].mozgoatlag[params.m2].datum[k].ev - 2000;
            if (ev<0) continue;
            if (k+7>=reszvenyek[i].mozgoatlag[params.m2].atlag.size()) continue;

            itrCnt2++;

            /// törés2

            float a1 = reszvenyek[i].mozgoatlag[params.m2].atlag[k-2];
            float a2 = reszvenyek[i].mozgoatlag[params.m2].atlag[k-1];
            float a3 = reszvenyek[i].mozgoatlag[params.m2].atlag[k-0];
            float b1 = a2-a1, b2 = a3-a2;

            int im1 = params.m2-params.m1;
            int im3 = params.m2-params.m3;

            bool V1 = false, V2 = true;
            /// V1
            if (V1){
                if (!params.toresAlatt){
                    if (b2<=0 || b1<=0 || b1<b2) continue;
                    if (b1-b2<=a2*params.tores) continue;
                } else {
                    if (b2>0 || b1>0 || b1>b2) continue;
                    if (b1-b2<a2*params.tores) continue;
                }
            }

            /// V2
            if (V2){
                //if ((b2<=0 || b1<=0 || b1<b2) && (b2>0 || b1>0 || b1>b2)) continue;
                if (!params.toresAlatt){
                    if ((b2<=0 || b1<=0 || b1<b2)) continue;
                    if (b1-b2<=a2*params.tores) continue;
                } else {
                    if ((b2>0 || b1>0 || b1>b2)) continue;
                    if (b1-b2>=a2*params.tores) continue;
                }
            }

            if (k+im3-1<0) continue;

            /// mozgó átlagok közötti növekvő sorrend check
            float ma1=reszvenyek[i].mozgoatlag[params.m1].atlag[k+im1];
            float ma2=reszvenyek[i].mozgoatlag[params.m2].atlag[k+0];
            float ma3=reszvenyek[i].mozgoatlag[params.m3].atlag[k+im3];

            if (params.ms==0) {if (!chkSorrend(ma3,ma2,ma1)) continue;}
            else if (params.ms==1) {if (!chkSorrend(ma3,ma1,ma2)) continue;}
            else if (params.ms==2) {if (!chkSorrend(ma2,ma1,ma3)) continue;}
            else if (params.ms==3) {if (!chkSorrend(ma2,ma3,ma1)) continue;}
            else if (params.ms==4) {if (!chkSorrend(ma1,ma3,ma2)) continue;}
            else if (params.ms==5) {if (!chkSorrend(ma1,ma2,ma3)) continue;}

            /// mozgóátlagok maguk növekvő/csökkenő
            if (params.mi==0){
                if (ma3-reszvenyek[i].mozgoatlag[params.m3].atlag[k+im3-1] <0) continue;
                if (ma1-reszvenyek[i].mozgoatlag[params.m1].atlag[k+im1-1] <0) continue;
            }
            else if (params.mi==1){
                if (ma3-reszvenyek[i].mozgoatlag[params.m3].atlag[k+im3-1] >=0) continue;
                if (ma1-reszvenyek[i].mozgoatlag[params.m1].atlag[k+im1-1] <0) continue;
            }
            else if (params.mi==2){
                if (ma3-reszvenyek[i].mozgoatlag[params.m3].atlag[k+im3-1] <0) continue;
                if (ma1-reszvenyek[i].mozgoatlag[params.m1].atlag[k+im1-1] >=0) continue;
            }
            else if (params.mi==3){
                if (ma3-reszvenyek[i].mozgoatlag[params.m3].atlag[k+im3-1] >=0) continue;
                if (ma1-reszvenyek[i].mozgoatlag[params.m1].atlag[k+im1-1] >=0) continue;
            }


            itrCnt3++;

            if (reszvenyek[i].nev=="MA"){
                ///cout<<reszvenyek[i].mozgoatlag[params.m2].datum[k].toString()<<endl;
            }
            //cout<<reszvenyek[i].nev<<endl;

            ///cout<<score.alkalmakEvente.size()<<" "<<tores_cnt_per_year[ev][params.m2]<<endl;
            ///tores_cnt_per_year[ev][params.m2]++;
            score.alkalmakEvente[ev]++;
            ///continue;
            pelda.stockName=reszvenyek[i].nev; pelda.datum=reszvenyek[i].mozgoatlag[params.m2].datum[k];
            pelda.reszvenyIdx=i; pelda.mozgoatlagIdx=k;
            osszesPelda.push_back(pelda);
            osszesEset.push_back(reszvenyek[i].mozgoatlag[params.m2].datum[k]);
            //xnapja=0;

        }
    }
    ///cout<<"ITR: "<<itrCnt<<" "<<itrCnt2<<" "<<itrCnt3<<endl;

    if (itrCnt3>50000) score;
    clock_t midtime = clock();
    ///cout<<"BALMA "<<ert<<endl;
    ///return 0;

    ///set<Datum> datumok = convertToSet(osszesEset);
    sort(osszesPelda.begin(),osszesPelda.end());
    list<Pelda> osszesPeldaList;
    osszesPeldaList.insert(osszesPeldaList.end(),osszesPelda.begin(),osszesPelda.end());
    cout<<osszesPelda.size()<<" "<<osszesPeldaList.size()<<endl;


    vector<float> napiErtek;
    napiErtek.resize(osszesDatum.size(),100.0f);

    int ev = 2000;
    float tempSzum = 0, tempCnt = 0;
    vector<float> eviErtekek;
    vector<vector<float>> eviNapok;
    ///cout<<"B2ALMA "<<ert<<endl;
    clock_t midtime2 = clock();
///    if (ert==-1) score.osszesPelda=osszesPelda;

    float peldasNapCnt=0;
    for (int i=0; i<osszesDatum.size(); i++){
        ///if (i%100==0)cout<<"O"<<i<<endl;
        ///continue;
        if (i+1<napiErtek.size())
            napiErtek[i+1]=napiErtek[i];
        if (osszesDatum[i].ev>ev){
            ev=osszesDatum[i].ev;
            ///napiErtek[i+0]=100;
            ///napiErtek[i+1]=100;
            score.evVegiek[ev-2001]=eviErtekek.back();
            eviNapok.push_back(eviErtekek);
            eviErtekek.clear();
        }
        vector<float> ertekek;
        ///if (datumok.find(osszesDatum[i])==datumok.end()){eviErtekek.push_back(napiErtek[i]); continue;}
        ///for (int j=0; j<osszesPelda.size(); j++){
        bool vanPelda = false;
        while(osszesPeldaList.front().datum==osszesDatum[i]){
            vanPelda=true;
            ///if (osszesPelda[j].datum==osszesDatum[i]){
                int stockIdx = osszesPeldaList.front().reszvenyIdx;
                int mozgoAtlagIdx = osszesPeldaList.front().mozgoatlagIdx;
                if (mozgoAtlagIdx+3>= reszvenyek[stockIdx].mozgoatlag[params.m2].zaras.size()) cout<<"ZZZZZZZ"<<endl;
                bool felfele = params.buy;///reszvenyek[stockIdx].mozgoatlag[params.m2].atlag[mozgoAtlagIdx]>reszvenyek[stockIdx].mozgoatlag[params.m2].atlag[mozgoAtlagIdx-1];
                int aznapiVetel = params.adasVeteliNapok; /// MELYIK METÓDUS
                if (aznapiVetel==0){
                    float aznapiZaras = reszvenyek[stockIdx].mozgoatlag[params.m2].zaras[mozgoAtlagIdx];
                    float masnapiZaras = reszvenyek[stockIdx].mozgoatlag[params.m2].zaras[mozgoAtlagIdx+1];
                    if (felfele) ertekek.push_back(masnapiZaras/aznapiZaras-1.0f);
                    else ertekek.push_back(-(masnapiZaras/aznapiZaras-1.0f));
                } else if (aznapiVetel==1){
                    float masnapiZaras = reszvenyek[stockIdx].mozgoatlag[params.m2].zaras[mozgoAtlagIdx+1];
                    float harmadnapiZaras = reszvenyek[stockIdx].mozgoatlag[params.m2].zaras[mozgoAtlagIdx+2];
                    if (felfele) ertekek.push_back(harmadnapiZaras/masnapiZaras-1.0f);
                    else ertekek.push_back(-(harmadnapiZaras/masnapiZaras-1.0f));
                } else if (aznapiVetel==2) {
                    float harmadnapiZaras = reszvenyek[stockIdx].mozgoatlag[params.m2].zaras[mozgoAtlagIdx+2];
                    float negyednapZaras = reszvenyek[stockIdx].mozgoatlag[params.m2].zaras[mozgoAtlagIdx+3];
                    if (felfele) ertekek.push_back(negyednapZaras/harmadnapiZaras-1.0f);
                    else ertekek.push_back(-(negyednapZaras/harmadnapiZaras-1.0f));
                } else if (aznapiVetel==3){
                    float aznapiZaras = reszvenyek[stockIdx].mozgoatlag[params.m2].zaras[mozgoAtlagIdx];
                    float harmadnapiZaras = reszvenyek[stockIdx].mozgoatlag[params.m2].zaras[mozgoAtlagIdx+2];
                    if (felfele) ertekek.push_back(harmadnapiZaras/aznapiZaras-1.0f);
                    else ertekek.push_back(-(harmadnapiZaras/aznapiZaras-1.0f));
                } else if (aznapiVetel==4) {
                    float masnapiZaras = reszvenyek[stockIdx].mozgoatlag[params.m2].zaras[mozgoAtlagIdx+1];
                    float negyednapZaras = reszvenyek[stockIdx].mozgoatlag[params.m2].zaras[mozgoAtlagIdx+3];
                    if (felfele) ertekek.push_back(negyednapZaras/masnapiZaras-1.0f);
                    else ertekek.push_back(-(negyednapZaras/masnapiZaras-1.0f));
                } else if (aznapiVetel==5) {
                    float aznapiZaras = reszvenyek[stockIdx].mozgoatlag[params.m2].zaras[mozgoAtlagIdx];
                    float negyednapZaras = reszvenyek[stockIdx].mozgoatlag[params.m2].zaras[mozgoAtlagIdx+3];
                    if (felfele) ertekek.push_back(negyednapZaras/aznapiZaras-1.0f);
                    else ertekek.push_back(-(negyednapZaras/aznapiZaras-1.0f));
                }
            ///}
            osszesPeldaList.pop_front();
        }
        if (vanPelda) peldasNapCnt++;
        int oszto = ertekek.size();
        float foszto = oszto;
        score.egyNapMaximum=max(oszto,score.egyNapMaximum);
        ///cout<<score.egyNapMaximum<<endl;

        for (int j=0; j<oszto; j++){
            float tempSum = 0;
            if (i+1<napiErtek.size()){
                napiErtek[i+1]+=napiErtek[i]*ertekek[j]/foszto;
                tempSzum+=ertekek[j];
                tempSum+=ertekek[j];;
                tempCnt++;
            }
            score.atlagosNapiProfit+=tempSum/foszto;
        }
        eviErtekek.push_back(napiErtek[i]);
    }
    score.atlagosNapiProfit/=max(1.0f,peldasNapCnt);
    clock_t endtime=clock();
    score.evVegiek[24]=eviErtekek.back();
    eviNapok.push_back(eviErtekek);
    ///cout<<"CALMA "<<ert<<endl;
    score.atlagosProfit=tempSzum/max(1.0f,tempCnt);
    score.teljes=napiErtek;
    if (ert==-2)cout<<"TIME: "<<endtime-midtime2<<" "<<midtime2-midtime<<" "<<midtime-stime<<endl;
    return 0;
}

vector<Datum> getOsszesDatum(vector<Reszveny>& reszvenyek){
    vector<Datum> osszesDatum;
    int legtobbNap = 0, idx=0;
    for (int i=0;i<reszvenyek.size(); i++){
        if (reszvenyek[i].napok.size()>legtobbNap) {
            legtobbNap=reszvenyek[i].napok.size();
            idx=i;
        }
    }
    for (int i=0;i<reszvenyek[idx].napok.size(); i++){
        if (reszvenyek[idx].napok[i].datum.ev<2000) continue;
        osszesDatum.push_back(reszvenyek[idx].napok[i].datum);
    }

    return osszesDatum;
}

void runTest(vector<Reszveny>& reszvenyek, Parameterek& params, Score& score, vector<Datum>& osszesDatum, int ert){

}

vector<ReszvenyGPU> getReszvenyekGPU(vector<Reszveny>& reszvenyek,vector<Datum>& osszesDatum){
    vector<ReszvenyGPU> ret;
    //vector<float> atlag(osszesDatum.size(),0);
    const int MOZGO_CNT = 50;
    for (int i=0; i<reszvenyek.size(); i++){
        if (i==383) continue; /// WKE
        cout<<"I "<<i<<" "<<reszvenyek[i].nev<<endl;
        ReszvenyGPU rgpu;
        rgpu.nev=reszvenyek[i].nev;
        rgpu.mozgoatlagokAtlag.resize(MOZGO_CNT,vector<float>(osszesDatum.size()));
        rgpu.mozgoatlagokZaras.resize(osszesDatum.size());
        rgpu.mozgoatlagokDatum.resize(osszesDatum.size());
        for (int z=0; z<osszesDatum.size(); z++){
            rgpu.mozgoatlagokZaras[z]=-1;
        }

        rgpu.N=osszesDatum.size();
        for (int j=0; j<MOZGO_CNT; j++){
            ///cout<<"J "<<j<<endl;
            int midx = 0;
            //rgpu.mozgoatlagokAtlag[j].resize(osszesDatum.size());
            //rgpu.mozgoatlagokDatum[j].resize(osszesDatum.size());
            for (int z=0; z<osszesDatum.size(); z++){
                ///if (i==469 && j==20 && z==137) cout<<"PEKING"<<endl;
                //cout<<"Z "<<z<<endl;
                ///if (i==2 && j==25 && z==1425) cout<<"ALMA "<<reszvenyek[i].nev<<endl;
                while (reszvenyek[i].mozgoatlag[j].datum[midx]<osszesDatum[z]){
                        ///if (i==469 && j==20 && z==137) cout<<"PEKING2"<<endl;
                    //z--;
                    midx++;
                    //continue;
                }
                if (osszesDatum[z]<reszvenyek[i].mozgoatlag[j].datum[midx]){
                    ///if (i==469 && j==20 && z==137) cout<<"PEKING1"<<endl;
                    rgpu.mozgoatlagokAtlag[j][z]=-1;
                    rgpu.mozgoatlagokDatum[z]=osszesDatum[z].toInt();
                    continue;
                }
                ///if (i==2 && j==25 && z==1425) cout<<"ALMA2"<<endl;
                ///cout<<"ok2"<<endl;
                ///if (i==469 && j==20 && z==137) cout<<"PEKING3"<<endl;
                rgpu.mozgoatlagokAtlag[j][z]=reszvenyek[i].mozgoatlag[j].atlag[midx];
                rgpu.mozgoatlagokZaras[z]=reszvenyek[i].mozgoatlag[j].zaras[midx];
                rgpu.mozgoatlagokDatum[z]=reszvenyek[i].mozgoatlag[j].datum[midx].toInt();
                ///osszesDatum[z].toInt();
                ///if (reszvenyek[i].nev=="AMD" && j==0 && rgpu.mozgoatlagokDatum[z]==osszesDatum[z].toInt()) cout<<"OK"<<endl;
                ///if (reszvenyek[i].nev=="AMD" && j==0)cout<<(int)osszesDatum[z].toInt()<<" 1"<<endl;
                ///if (reszvenyek[i].nev=="AMD" && j==0)cout<<(int)rgpu.mozgoatlagokDatum[z]<<"   2"<<endl;
                ///if (reszvenyek[i].nev=="AMD" && j==0)cout<<(int)osszesDatum[z].toInt()-(int)rgpu.mozgoatlagokDatum[z]<<"   2"<<endl;
                midx++;
            }
        }
        ret.push_back(rgpu);
    }

    return ret;
}

int main(){
    clock_t fullT = clock();
    vector<string> reszvenyekFajlNeve = reszvenyekEleresiUtja("etoroStocks.txt","data");
    vector<Reszveny> reszvenyek = reszvenyekParhuzamosBetoltese(reszvenyekFajlNeve);
    vector<Datum> osszesDatum = getOsszesDatum(reszvenyek);
    clock_t time0 = clock();
    cout<<"DONE "<<osszesDatum.size()<<endl;
    vector<ReszvenyGPU> reszvenyekGPU = getReszvenyekGPU(reszvenyek, osszesDatum);
    ///reszvenyek.clear();
    ///cout<<reszvenyekGPU[1].nev<<endl;
    ///for (int i=0; i<reszvenyekGPU[1].mozgoatlagokDatum.size(); i++){
        ///if (osszesDatum[i].ev<2024) continue;
        ///cout<<osszesDatum[i].toInt()<<" 1"<<endl;
        ///cout<<reszvenyekGPU[1].mozgoatlagokDatum[i]<<" "<<reszvenyekGPU[1].mozgoatlagokZaras[i]<<"   1"<<endl;
    ///}
    ///cout<<osszesDatum.size()<<" "<<reszvenyekGPU[1].mozgoatlagokDatum.size()<<endl;
    ///ResGpu resGpu = getResGpu(reszvenyekGPU);
    clock_t time01 = clock();
    cout<<"DONE 2: "<<time01-time0<<endl;

    bool candleTest=false;
    if (candleTest){
        set<Eset> esetek1;
        set<Eset> esetek2;
        string abc = "abcdefghijklmnopqrstuvwxyz";
        const int napszam=4;
        vector<Par> parok(napszam*4);
        stringstream ss;
        for (int i=0; i<napszam*4; i++)
            ss<<'a';
        string str=ss.str();
        for (int i=0; i<reszvenyek.size(); i++){
            float zarasok[napszam];
            float maxok[napszam];
            float minek[napszam];
            float nyitasok[napszam];
            for (int j=0; j+2<reszvenyek[i].napok.size(); j++){
                for (int k=0; k<napszam-1; k++){
                    zarasok[k]=zarasok[k+1];
                    maxok[k]=maxok[k+1];
                    minek[k]=minek[k+1];
                    nyitasok[k]=nyitasok[k+1];
                }
                zarasok[napszam-1] = reszvenyek[i].napok[j].zaras;
                maxok[napszam-1] = reszvenyek[i].napok[j].maximum;
                minek[napszam-1] = reszvenyek[i].napok[j].minimum;
                nyitasok[napszam-1] = reszvenyek[i].napok[j].nyitas;
                if (j<napszam-1) continue;
                if (reszvenyek[i].napok[j].datum.ev<2020) continue;

                set<float> ertekek;
                for (int k=0; k<napszam; k++){
                    Par par1; par1.ertek=zarasok[k];    par1.karakter=abc[0+k*4]; parok[0+k*4] = par1; ertekek.insert(zarasok[k]);
                    Par par2; par2.ertek=maxok[k];      par2.karakter=abc[1+k*4]; parok[1+k*4] = par2; ertekek.insert(maxok[k]);
                    Par par3; par3.ertek=minek[k];      par3.karakter=abc[2+k*4]; parok[2+k*4] = par3; ertekek.insert(minek[k]);
                    Par par4; par4.ertek=nyitasok[k];   par4.karakter=abc[3+k*4]; parok[3+k*4] = par4; ertekek.insert(nyitasok[k]);
                }
                if (ertekek.size()<(napszam*4*0.7f)) continue;
                sort(parok.begin(), parok.end());
                //string str="aaaaaaaaaaaaaaaaaaaa";
                for (int k=0; k<napszam*4; k++){
                    str[k]=parok[k].karakter;
                }
                float kovetkezoZaras = reszvenyek[i].napok[j+1].zaras;
                float aztKovetoZaras = reszvenyek[i].napok[j+2].zaras;
                Eset eset1; eset1.charChain=str; eset1.osszesEset=1;
                eset1.prod = kovetkezoZaras/zarasok[napszam-1]; eset1.szum = kovetkezoZaras/zarasok[napszam-1]-1.0f;
                if (eset1.prod>=1.0f) eset1.pozitivEset=1;
                Eset eset2; eset2.charChain=str; eset2.osszesEset=1;
                eset2.prod = aztKovetoZaras/zarasok[napszam-1]; eset2.szum = aztKovetoZaras/zarasok[napszam-1]-1.0f;
                if (eset2.prod>=1.0f) eset2.pozitivEset=1;
                set<Eset>::iterator it1=esetek1.find(eset1);
                set<Eset>::iterator it2=esetek2.find(eset2);
                if (it1==esetek1.end()){
                    esetek1.insert(eset1);
                }
                else {
                    Eset temp = *it1;
                    temp.osszesEset++;
                    if (eset1.prod>=1.0f)
                        temp.pozitivEset++;
                    temp.prod*=eset1.prod;
                    temp.szum+=eset1.prod-1.0f;
                    esetek1.erase(it1);
                    esetek1.insert(temp);
                }
                if (it2==esetek2.end()){
                    esetek2.insert(eset2);
                }
                else {
                    Eset temp = *it2;
                    temp.osszesEset++;
                    if (eset2.prod>=1.0f)
                        temp.pozitivEset++;
                    temp.prod*=eset2.prod;
                    temp.szum+=eset2.prod-1.0f;
                    esetek2.erase(it2);
                    esetek2.insert(temp);
                }
            }
            cout<<"ES1: "<<esetek1.size()<<", ES2: "<<esetek2.size()<<endl;
        }

        ofstream ofile1("esetek1_4_2020.txt");
        for(Eset eset: esetek1){
            ofile1<<eset.charChain<<" "<<eset.osszesEset<<" "<<eset.pozitivEset<<" "<<eset.pozitivEset/eset.osszesEset<<" "<<eset.szum<<" "<<eset.prod<<endl;
        }
        ofile1.close();
        ofstream ofile2("esetek2_4_2020.txt");
        for(Eset eset: esetek2){
            ofile2<<eset.charChain<<" "<<eset.osszesEset<<" "<<eset.pozitivEset<<" "<<eset.pozitivEset/eset.osszesEset<<" "<<eset.szum<<" "<<eset.prod<<endl;
        }
        ofile2.close();

        return 0;
    }
    ///reszvenyek.clear();


    bool pozitivTest=false;
    if (pozitivTest){
        const int napszam=10;
        float osszesEset[napszam];
        float pozitivEset[napszam];
        string abc = "abcdefghijklmnopqrstuvwxyz";
        vector<Par> parok(napszam*4);
        stringstream ss;
        for (int i=0; i<napszam*4; i++)
            ss<<'a';
        string str=ss.str();
        for (int i=0; i<reszvenyek.size(); i++){
            float zarasok[napszam];
            float maxok[napszam];
            float minek[napszam];
            float nyitasok[napszam];
            cout<<i<<endl;
            for (int j=0; j+napszam+1<reszvenyek[i].napok.size(); j++){
                    ///cout<<"alma "<<j<<endl;
                for (int k=0; k<napszam-1; k++){
                    zarasok[k]=zarasok[k+1];
                    maxok[k]=maxok[k+1];
                    minek[k]=minek[k+1];
                    nyitasok[k]=nyitasok[k+1];
                }
                zarasok[napszam-1] = reszvenyek[i].napok[j].zaras;
                maxok[napszam-1] = reszvenyek[i].napok[j].maximum;
                minek[napszam-1] = reszvenyek[i].napok[j].minimum;
                nyitasok[napszam-1] = reszvenyek[i].napok[j].nyitas;
                if (j<napszam-1) continue;


                set<float> ertekek;
                for (int k=0; k<napszam; k++){
                    Par par1; par1.ertek=zarasok[k];    par1.karakter=abc[0+k*4]; parok[0+k*4] = par1; ertekek.insert(zarasok[k]);
                    Par par2; par2.ertek=maxok[k];      par2.karakter=abc[1+k*4]; parok[1+k*4] = par2; ertekek.insert(maxok[k]);
                    Par par3; par3.ertek=minek[k];      par3.karakter=abc[2+k*4]; parok[2+k*4] = par3; ertekek.insert(minek[k]);
                    Par par4; par4.ertek=nyitasok[k];   par4.karakter=abc[3+k*4]; parok[3+k*4] = par4; ertekek.insert(nyitasok[k]);
                }
                if (ertekek.size()<(napszam*4*0.7f)) continue;

                ///sort(parok.begin(), parok.end());
                //string str="aaaaaaaaaaaaaaaaaaaa";
                for (int k=0; k<napszam*4; k++){
                    str[k]=parok[k].karakter;
                }
                for (int k=0; k<napszam; k++){
                    osszesEset[k]++;
                    //if (reszvenyek[i].napok[j+k+1].zaras > reszvenyek[i].napok[j].zaras)
                    pozitivEset[k]+= (reszvenyek[i].napok[j+k+1].zaras/reszvenyek[i].napok[j].zaras)-1.0f;
                }
            }
            for (int k=0; k<napszam; k++){
                cout<<pozitivEset[k]/osszesEset[k]<<" ";
            }
            cout<<endl;
            //cout<<"ES1: "<<esetek1.size()<<", ES2: "<<esetek2.size()<<endl;
        }

        return 0;
    }
    reszvenyek.clear();

    cout<<reszvenyekGPU[0].mozgoatlagokDatum[6037]<<" "<<reszvenyekGPU[0].mozgoatlagokDatum[6047]<<endl;
    cout<<reszvenyekGPU[0].mozgoatlagokDatum[6037+126]<<" "<<reszvenyekGPU[0].mozgoatlagokDatum[6047+126]<<endl;
    bool hasonloTestMin=false;
    if (hasonloTestMin){ /// 6037 - 6182,
        int istart = 6037, iend = istart+145;
        int hossz = iend-istart;
        int kornyezet = 10;
        int eltolas = 10;
        int vizsgaltIdo = 10;
        vector<vector<float>> hasonlosagok(reszvenyekGPU.size());
        for (int i=0; i<reszvenyekGPU.size(); i++){
            cout<<"I: "<<i<<endl;
            vector<float> normaltArfolyam;
            for (int j=istart+eltolas; j<=iend-vizsgaltIdo; j++){
                normaltArfolyam.push_back(reszvenyekGPU[i].mozgoatlagokZaras[j]/reszvenyekGPU[i].mozgoatlagokZaras[istart+eltolas]);
                ///cout<<reszvenyekGPU[i].mozgoatlagokDatum[j]<<" "<<j<<endl;
            }
            ///cout<<normaltArfolyam.size()<<endl;
            for (int j=0; j<reszvenyekGPU.size(); j++){
                vector<float> normaltMasikArfolyam;
                float minSzum = 100000;
                for (int z=0; z<1; z++){
                    normaltMasikArfolyam.clear();
                    for (int k=istart+z; k<=iend-vizsgaltIdo-eltolas+z; k++){
                        normaltMasikArfolyam.push_back(reszvenyekGPU[j].mozgoatlagokZaras[k]/reszvenyekGPU[j].mozgoatlagokZaras[istart+z]);
                    }
                    ///cout<<normaltMasikArfolyam.size()<<endl;
                    float szum = 0;
                    for (int k=0; k<normaltMasikArfolyam.size(); k++){
                        szum+=fabs(normaltArfolyam[k]-normaltMasikArfolyam[k]);
                    }
                    minSzum = min(minSzum,szum);
                }
                if (i==j) hasonlosagok[i].push_back(42);
                else hasonlosagok[i].push_back(minSzum);
            }
        }
        ofstream ofileHas("hasonlok_101.txt");
        for (int i=0;i<hasonlosagok.size(); i++) ofileHas<<reszvenyekGPU[i].nev<<" ";
        ofileHas<<endl;
        for (int i=0;i<hasonlosagok.size(); i++){
            ofileHas<<reszvenyekGPU[i].nev<<" ";
            for (int j=0;j<hasonlosagok[i].size(); j++){
                ofileHas<<hasonlosagok[i][j]<<" ";
            }
            ofileHas<<endl;
        }
        ofileHas.close();

        for (int i=0; i<reszvenyekGPU.size(); i++){
            vector<float> minimumok(3,10000);
            vector<float> minIdx(3,-1);
            for (int j=0; j<reszvenyekGPU.size(); j++){
                if (hasonlosagok[i][j]<minimumok[2]){
                    minimumok[0]=minimumok[1];
                    minimumok[1]=minimumok[2];
                    minimumok[2]=hasonlosagok[i][j];
                    minIdx[0]=minIdx[1];
                    minIdx[1]=minIdx[2];
                    minIdx[2]=j;
                } else if (hasonlosagok[i][j]<minimumok[1]){
                    minimumok[0]=minimumok[1];
                    minimumok[1]=hasonlosagok[i][j];
                    minIdx[0]=minIdx[1];
                    minIdx[1]=j;
                } else if (hasonlosagok[i][j]<minimumok[0]){
                    minimumok[0]=hasonlosagok[i][j];
                    minIdx[0]=j;
                }
            }
            float a = reszvenyekGPU[minIdx[0]].mozgoatlagokZaras[iend-vizsgaltIdo]-reszvenyekGPU[minIdx[0]].mozgoatlagokZaras[iend-vizsgaltIdo-eltolas];
            float b = reszvenyekGPU[minIdx[1]].mozgoatlagokZaras[iend-vizsgaltIdo]-reszvenyekGPU[minIdx[1]].mozgoatlagokZaras[iend-vizsgaltIdo-eltolas];
            float c = reszvenyekGPU[minIdx[2]].mozgoatlagokZaras[iend-vizsgaltIdo]-reszvenyekGPU[minIdx[2]].mozgoatlagokZaras[iend-vizsgaltIdo-eltolas];
            float hm[3];
            if (a<0) hm[0] = reszvenyekGPU[i].mozgoatlagokZaras[iend-vizsgaltIdo+eltolas]/reszvenyekGPU[i].mozgoatlagokZaras[iend-vizsgaltIdo];
            //else
            hm[0] = reszvenyekGPU[i].mozgoatlagokZaras[iend-vizsgaltIdo+eltolas]/reszvenyekGPU[i].mozgoatlagokZaras[iend-vizsgaltIdo];
            if (a<0) hm[0]=1.0f+(1.0f-hm[0]);
            hm[1] = reszvenyekGPU[i].mozgoatlagokZaras[iend-vizsgaltIdo+eltolas]/reszvenyekGPU[i].mozgoatlagokZaras[iend-vizsgaltIdo];
            if (b<0) hm[1]=1.0f+(1.0f-hm[1]);
            hm[2] = reszvenyekGPU[i].mozgoatlagokZaras[iend-vizsgaltIdo+eltolas]/reszvenyekGPU[i].mozgoatlagokZaras[iend-vizsgaltIdo];
            if (c<0) hm[2]=1.0f+(1.0f-hm[2]);
            cout<<reszvenyekGPU[i].nev<<" ";
            for (int k=0; k<3;k++) cout<<reszvenyekGPU[minIdx[k]].nev<<" "<<minimumok[k]<<" "<<hm[k]<<" ";
            cout<<endl;
        }


        return 0;
    }


    bool hasonloTestMinTobbNap2=true;
    if (hasonloTestMinTobbNap2){ /// 6037 - 6182,
        ofstream ofileHas("hasonlok_202_o.txt");
        for (int zzz=4000; zzz<6050; zzz++){
            ofileHas<<zzz<<" "<<reszvenyekGPU.size()<<endl;
            cout<<"Z: "<<zzz<<endl;
            int istart = zzz, iend = istart+145;
            int hossz = iend-istart;
            int kornyezet = 10;
            int eltolas = 10;
            int vizsgaltIdo = 10;
            int hibasErtek = 420;
            vector<vector<float>> hasonlosagok(reszvenyekGPU.size());
            for (int i=0; i<reszvenyekGPU.size(); i++){
                ///cout<<"I: "<<i<<endl;
                vector<float> normaltArfolyam;
                bool baj = false;

                float arfMin = 100000, arfMax = 0;

                for (int j=istart+eltolas; j<=iend-vizsgaltIdo; j++){
                    arfMax=max(reszvenyekGPU[i].mozgoatlagokZaras[j],arfMax);
                    arfMin=min(reszvenyekGPU[i].mozgoatlagokZaras[j],arfMin);
                }
                baj = false;
                ///cout<<"OK1"<<endl;
                for (int j=istart+eltolas; j<=iend-vizsgaltIdo; j++){
                    if (reszvenyekGPU[i].mozgoatlagokZaras[j]==-1) {
                        baj=true; break;
                    }
                    float aznapiErtek = reszvenyekGPU[i].mozgoatlagokZaras[j];
                    float normalizaltErtek = 0;
                    if (arfMax!=arfMin) normalizaltErtek = (aznapiErtek-arfMin)/(arfMax-arfMin);
                    //normalizaltErtek = reszvenyekGPU[i].mozgoatlagokZaras[j]/reszvenyekGPU[i].mozgoatlagokZaras[istart+eltolas];
                    else {baj=true; break;}
                    normaltArfolyam.push_back(normalizaltErtek);
                    ///cout<<reszvenyekGPU[i].mozgoatlagokDatum[j]<<" "<<j<<endl;
                }

                for (int j=istart+eltolas; false && j<=iend-vizsgaltIdo; j++){
                    if (reszvenyekGPU[i].mozgoatlagokZaras[j]==-1) {
                        baj=true; break;
                    }
                    normaltArfolyam.push_back(reszvenyekGPU[i].mozgoatlagokZaras[j]/reszvenyekGPU[i].mozgoatlagokZaras[istart+eltolas]);
                    ///cout<<reszvenyekGPU[i].mozgoatlagokDatum[j]<<" "<<j<<endl;
                }

                if (baj){
                    for (int j=0; j<reszvenyekGPU.size(); j++)
                        hasonlosagok[i].push_back(hibasErtek);
                    continue;
                }
                ///cout<<normaltArfolyam.size()<<endl;
                for (int j=0; j<reszvenyekGPU.size(); j++){
                    baj = false;
                    vector<float> normaltMasikArfolyam;
                    float minSzum = 100000;

                    normaltMasikArfolyam.clear();
                    float arfMin2 = 100000, arfMax2 = 0;

                    for (int k=istart+0; k<=iend-vizsgaltIdo-eltolas+0; k++){
                        arfMax2=max(reszvenyekGPU[j].mozgoatlagokZaras[k],arfMax2);
                        arfMin2=min(reszvenyekGPU[j].mozgoatlagokZaras[k],arfMin2);
                    }
                    for (int k=istart+0; k<=iend-vizsgaltIdo-eltolas+0; k++){
                        if (reszvenyekGPU[j].mozgoatlagokZaras[k]==-1){baj=true; break;}
                        ///normaltMasikArfolyam.push_back(reszvenyekGPU[j].mozgoatlagokZaras[k]/reszvenyekGPU[j].mozgoatlagokZaras[istart+0]);
                        float aznapiErtek = reszvenyekGPU[j].mozgoatlagokZaras[k];
                        float normalizaltErtek = 0;
                        if (arfMax2!=arfMin2) normaltMasikArfolyam.push_back((aznapiErtek-arfMin2)/(arfMax2-arfMin2));
                        else {baj=true; break;}
                    }
                    if (baj){
                        hasonlosagok[i].push_back(hibasErtek);
                        continue;
                    }
                    ///cout<<normaltMasikArfolyam.size()<<endl;
                    float szum = 0;
                    for (int k=0; k<normaltMasikArfolyam.size(); k++){
                        szum+=fabs(normaltArfolyam[k]-normaltMasikArfolyam[k]);
                    }

                    minSzum = min(minSzum,szum);

                    if (i==j) hasonlosagok[i].push_back(hibasErtek);
                    else hasonlosagok[i].push_back(minSzum);
                }
            }
            /*
            ofstream ofileHas("hasonlok_101.txt");
            for (int i=0;i<hasonlosagok.size(); i++) ofileHas<<reszvenyekGPU[i].nev<<" ";
            ofileHas<<endl;
            for (int i=0;i<hasonlosagok.size(); i++){
                ofileHas<<reszvenyekGPU[i].nev<<" ";
                for (int j=0;j<hasonlosagok[i].size(); j++){
                    ofileHas<<hasonlosagok[i][j]<<" ";
                }
                ofileHas<<endl;
            }
            ofileHas.close();
            */


            for (int i=0; i<reszvenyekGPU.size(); i++){
                vector<float> minimumok(3,10000);
                vector<float> minIdx(3,-1);
                for (int j=0; j<reszvenyekGPU.size(); j++){
                    if (hasonlosagok[i][j]<minimumok[2]){
                        minimumok[0]=minimumok[1];
                        minimumok[1]=minimumok[2];
                        minimumok[2]=hasonlosagok[i][j];
                        minIdx[0]=minIdx[1];
                        minIdx[1]=minIdx[2];
                        minIdx[2]=j;
                    } else if (hasonlosagok[i][j]<minimumok[1]){
                        minimumok[0]=minimumok[1];
                        minimumok[1]=hasonlosagok[i][j];
                        minIdx[0]=minIdx[1];
                        minIdx[1]=j;
                    } else if (hasonlosagok[i][j]<minimumok[0]){
                        minimumok[0]=hasonlosagok[i][j];
                        minIdx[0]=j;
                    }
                }
                float a = reszvenyekGPU[minIdx[0]].mozgoatlagokZaras[iend-vizsgaltIdo]-reszvenyekGPU[minIdx[0]].mozgoatlagokZaras[iend-vizsgaltIdo-eltolas];
                float b = reszvenyekGPU[minIdx[1]].mozgoatlagokZaras[iend-vizsgaltIdo]-reszvenyekGPU[minIdx[1]].mozgoatlagokZaras[iend-vizsgaltIdo-eltolas];
                float c = reszvenyekGPU[minIdx[2]].mozgoatlagokZaras[iend-vizsgaltIdo]-reszvenyekGPU[minIdx[2]].mozgoatlagokZaras[iend-vizsgaltIdo-eltolas];
                float hm[3];
                if (a<0) hm[0] = reszvenyekGPU[i].mozgoatlagokZaras[iend-vizsgaltIdo+eltolas]/reszvenyekGPU[i].mozgoatlagokZaras[iend-vizsgaltIdo];
                //else
                hm[0] = reszvenyekGPU[i].mozgoatlagokZaras[iend-vizsgaltIdo+eltolas]/reszvenyekGPU[i].mozgoatlagokZaras[iend-vizsgaltIdo];
                if (a<0)
                    hm[0]=1.0f+(1.0f-hm[0]);
                hm[1] = reszvenyekGPU[i].mozgoatlagokZaras[iend-vizsgaltIdo+eltolas]/reszvenyekGPU[i].mozgoatlagokZaras[iend-vizsgaltIdo];
                if (b<0)
                    hm[1]=1.0f+(1.0f-hm[1]);
                hm[2] = reszvenyekGPU[i].mozgoatlagokZaras[iend-vizsgaltIdo+eltolas]/reszvenyekGPU[i].mozgoatlagokZaras[iend-vizsgaltIdo];
                if (c<0)
                    hm[2]=1.0f+(1.0f-hm[2]);
                ofileHas<<reszvenyekGPU[i].nev<<" ";
                for (int k=0; k<3;k++) ofileHas<<reszvenyekGPU[minIdx[k]].nev<<" "<<minimumok[k]<<" "<<hm[k]<<" ";
                ofileHas<<endl;
            }
        }


        return 0;
    }


    bool hasonloTestMinTobbNap=false;
    if (hasonloTestMinTobbNap){ /// 6037 - 6182,
        int start = 4000;
        vector<vector<float>> vegsoErtekek(6050-start,vector<float>(15));
        for (int zzz=start; zzz<6050; zzz++){
            cout<<"Z: "<<zzz<<endl;
            int istart = zzz, iend = istart+145;
            int hossz = iend-istart;
            int kornyezet = 10;
            int eltolas = 10;
            int vizsgaltIdo = 10;
            vector<vector<float>> hasonlosagok(reszvenyekGPU.size());
            for (int i=0; i<reszvenyekGPU.size(); i++){
                ///cout<<"I: "<<i<<endl;
                vector<float> normaltArfolyam;
                float arfMin = 100000, arfMax = 0;

                for (int j=istart+eltolas; j<=iend-vizsgaltIdo; j++){
                    arfMax=max(reszvenyekGPU[i].mozgoatlagokZaras[j],arfMax);
                    arfMin=min(reszvenyekGPU[i].mozgoatlagokZaras[j],arfMin);
                }
                bool baj = false;
                ///cout<<"OK1"<<endl;
                for (int j=istart+eltolas; j<=iend-vizsgaltIdo; j++){
                    float aznapiErtek = reszvenyekGPU[i].mozgoatlagokZaras[j];
                    float normalizaltErtek = 0;
                    if (arfMax!=arfMin) normalizaltErtek = (aznapiErtek-arfMin)/(arfMax-arfMin);
                    //normalizaltErtek = reszvenyekGPU[i].mozgoatlagokZaras[j]/reszvenyekGPU[i].mozgoatlagokZaras[istart+eltolas];
                    else {baj=true; break;}
                    normaltArfolyam.push_back(normalizaltErtek);
                    ///cout<<reszvenyekGPU[i].mozgoatlagokDatum[j]<<" "<<j<<endl;
                }
                ///cout<<"OK3"<<endl;
                if (baj) {
                    for (int j=0; j<reszvenyekGPU.size(); j++){
                        hasonlosagok[i].push_back(420);
                    }
                    break;
                }
                ///cout<<"OK2"<<endl;
                ///cout<<normaltArfolyam.size()<<endl;
                for (int j=0; j<reszvenyekGPU.size(); j++){
                    baj=false;
                    vector<float> normaltMasikArfolyam;
                    float minSzum = 100000;
                    for (int z=0; z<1; z++){
                        normaltMasikArfolyam.clear();
                        float arfMin2 = 100000, arfMax2 = 0;
                        for (int k=istart+z; k<=iend-vizsgaltIdo-eltolas+z; k++){
                            arfMax2=max(reszvenyekGPU[j].mozgoatlagokZaras[k],arfMax2);
                            arfMin2=min(reszvenyekGPU[j].mozgoatlagokZaras[k],arfMin2);
                        }
                        for (int k=istart+z; k<=iend-vizsgaltIdo-eltolas+z; k++){
                            ///normaltMasikArfolyam.push_back(reszvenyekGPU[j].mozgoatlagokZaras[k]/reszvenyekGPU[j].mozgoatlagokZaras[istart+z]);
                            float aznapiErtek = reszvenyekGPU[j].mozgoatlagokZaras[k];
                            float normalizaltErtek = 0;
                            if (arfMax2!=arfMin2) normaltMasikArfolyam.push_back((aznapiErtek-arfMin2)/(arfMax2-arfMin2));
                            else {baj=true; break;}
                        }
                        if (baj) {
                            ///cout<<"baj ";
                            minSzum=420;
                            continue;
                        }
                        ///cout<<normaltMasikArfolyam.size()<<endl;
                        float szum = 0;
                        for (int k=0; k<normaltMasikArfolyam.size(); k++){
                            szum+=fabs(normaltArfolyam[k]-normaltMasikArfolyam[k]);
                            if (szum==0){
                                ///cout<<"sz "<<normaltArfolyam[k]<<" "<<normaltMasikArfolyam[k]<<endl;
                            }
                        }
                        ///if (szum==0) cout<<normaltMasikArfolyam.size()<<endl;
                        ///cout<<",m: "<<minSzum<<": "<<szum<<" ";
                        if (szum==0) szum=420;
                        minSzum = min(minSzum,szum);
                    }
                    if (i==j) hasonlosagok[i].push_back(420);
                    else hasonlosagok[i].push_back(minSzum);
                    ///cout<<minSzum<<" "<<hasonlosagok[i].back()<<" "<<baj<<endl;
                }
            }

            /**
            ofstream ofileHas("hasonlok_101.txt");
            for (int i=0;i<hasonlosagok.size(); i++) ofileHas<<reszvenyekGPU[i].nev<<" ";
            ofileHas<<endl;
            for (int i=0;i<hasonlosagok.size(); i++){
                ofileHas<<reszvenyekGPU[i].nev<<" ";
                for (int j=0;j<hasonlosagok[i].size(); j++){
                    ofileHas<<hasonlosagok[i][j]<<" ";
                }
                ofileHas<<endl;
            }
            ofileHas.close();
            */

            vector<vector<Par2>> parok(reszvenyekGPU.size());
            for (int i=0; i<reszvenyekGPU.size(); i++){
                vector<float> minimumok(3,100000);
                vector<float> minIdx(3,-1);
                for (int j=0; j<reszvenyekGPU.size(); j++){
                    if (hasonlosagok[i][j]==420) continue;
                    if (hasonlosagok[i][j]<minimumok[2]){
                        minimumok[0]=minimumok[1];
                        minimumok[1]=minimumok[2];
                        minimumok[2]=hasonlosagok[i][j];
                        minIdx[0]=minIdx[1];
                        minIdx[1]=minIdx[2];
                        minIdx[2]=j;
                    } else if (hasonlosagok[i][j]<minimumok[1]){
                        minimumok[0]=minimumok[1];
                        minimumok[1]=hasonlosagok[i][j];
                        minIdx[0]=minIdx[1];
                        minIdx[1]=j;
                    } else if (hasonlosagok[i][j]<minimumok[0]){
                        minimumok[0]=hasonlosagok[i][j];
                        minIdx[0]=j;
                    }
                }
                if (minIdx[0] < 0 || minIdx[1] < 0 || minIdx[2] < 0){
                    Par2 par; par.ertek=420; par.sorszam=1;
                    parok[i].push_back(par);
                    continue;
                }

                float a = reszvenyekGPU[minIdx[0]].mozgoatlagokZaras[iend-vizsgaltIdo]-reszvenyekGPU[minIdx[0]].mozgoatlagokZaras[iend-vizsgaltIdo-eltolas];
                float b = reszvenyekGPU[minIdx[1]].mozgoatlagokZaras[iend-vizsgaltIdo]-reszvenyekGPU[minIdx[1]].mozgoatlagokZaras[iend-vizsgaltIdo-eltolas];
                float c = reszvenyekGPU[minIdx[2]].mozgoatlagokZaras[iend-vizsgaltIdo]-reszvenyekGPU[minIdx[2]].mozgoatlagokZaras[iend-vizsgaltIdo-eltolas];
                float hm[3];
                for (int j=0; j<reszvenyekGPU.size(); j++){
                    if (i==j) continue;
                    float ui = reszvenyekGPU[j].mozgoatlagokZaras[iend-vizsgaltIdo]-reszvenyekGPU[j].mozgoatlagokZaras[iend-vizsgaltIdo-eltolas];
                    float ert = reszvenyekGPU[i].mozgoatlagokZaras[iend-vizsgaltIdo+eltolas]/reszvenyekGPU[i].mozgoatlagokZaras[iend-vizsgaltIdo];
                    ///if (ui>0)
                        ///ert = 1.0f/ert;
                    Par2 par; par.ertek=hasonlosagok[i][j]; par.sorszam=ert;
                    parok[i].push_back(par);
                }
                /**
                if (a<0) hm[0] = reszvenyekGPU[i].mozgoatlagokZaras[iend-vizsgaltIdo+eltolas]/reszvenyekGPU[i].mozgoatlagokZaras[iend-vizsgaltIdo];
                //else
                hm[0] = reszvenyekGPU[i].mozgoatlagokZaras[iend-vizsgaltIdo+eltolas]/reszvenyekGPU[i].mozgoatlagokZaras[iend-vizsgaltIdo];
                if (a<0) hm[0]=1.0f/hm[0];
                hm[1] = reszvenyekGPU[i].mozgoatlagokZaras[iend-vizsgaltIdo+eltolas]/reszvenyekGPU[i].mozgoatlagokZaras[iend-vizsgaltIdo];
                if (b<0) hm[1]=1.0f/hm[1];
                hm[2] = reszvenyekGPU[i].mozgoatlagokZaras[iend-vizsgaltIdo+eltolas]/reszvenyekGPU[i].mozgoatlagokZaras[iend-vizsgaltIdo];
                if (c<0) hm[2]=1.0f/hm[2];
                cout<<reszvenyekGPU[i].nev<<" ";
                for (int k=0; k<3;k++) cout<<reszvenyekGPU[minIdx[k]].nev<<" "<<minimumok[k]<<" "<<hm[k]<<" ";
                    */
                ///cout<<endl;
            }

            for (int i=0; i<parok.size(); i++) {
                sort(parok[i].begin(), parok[i].end());
                ///reverse(parok[i].begin(), parok[i].end());
            }
            vector<vector<Par2>> topParok(3);
            for (int i=0; i<parok.size(); i++){
                topParok[0].push_back(parok[i][0]);
                topParok[1].push_back(parok[i][1]);
                topParok[2].push_back(parok[i][2]);
            }
            for (int i=0; i<topParok.size(); i++) {
                sort(topParok[i].begin(), topParok[i].end());
                ///reverse(topParok[i].begin(), topParok[i].end());
            }
            float topok[5]={200,100,40,10,5};
            for (int k=0; k<5; k++){
                float s1=0, s2=0, s3=0;
                for (int i=0; i<topok[k]; i++){
                    s1+=topParok[0][i].sorszam;
                    s2+=topParok[1][i].sorszam;
                    s3+=topParok[2][i].sorszam;
                }
                vegsoErtekek[zzz-start][0+k]=s1/topok[k]; vegsoErtekek[zzz-start][5+k]=s2/topok[k]; vegsoErtekek[zzz-start][10+k]=s3/topok[k];
            }

        }
        ofstream ofileVegso("hasonloakVegso_4000-6050_5_fullBuy3.txt");
        for (int i=0; i<vegsoErtekek.size(); i++){
            ofileVegso<<reszvenyekGPU[0].mozgoatlagokDatum[start+i]<<" "<<start+i<<" ";
            for (int k=0; k<15; k++) ofileVegso<<vegsoErtekek[i][k]<<" ";
            ofileVegso<<endl;
        }
        ofileVegso.close();
        return 0;
    }


    bool hasonloTestMax=false;
    if (hasonloTestMax){ /// 6037 - 6182,
        system("cls");
        int istart = 6037, iend = istart+145;
        int hossz = iend-istart;
        int kornyezet = 10;
        int eltolas = 10;
        int vizsgaltIdo = 10;
        vector<vector<float>> hasonlosagok(reszvenyekGPU.size());
        for (int i=0; i<reszvenyekGPU.size(); i++){
            cout<<"I: "<<i<<endl;
            vector<float> normaltArfolyam;
            for (int j=istart+eltolas; j<=iend-vizsgaltIdo; j++){
                normaltArfolyam.push_back(reszvenyekGPU[i].mozgoatlagokZaras[j]/reszvenyekGPU[i].mozgoatlagokZaras[istart+eltolas]);
                ///cout<<reszvenyekGPU[i].mozgoatlagokDatum[j]<<" "<<j<<endl;
            }
            ///cout<<normaltArfolyam.size()<<endl;
            for (int j=0; j<reszvenyekGPU.size(); j++){
                vector<float> normaltMasikArfolyam;
                float maxSzum = 0;
                for (int z=0; z<1; z++){
                    normaltMasikArfolyam.clear();
                    for (int k=istart+z; k<=iend-vizsgaltIdo-eltolas+z; k++){
                        normaltMasikArfolyam.push_back(reszvenyekGPU[j].mozgoatlagokZaras[k]/reszvenyekGPU[j].mozgoatlagokZaras[istart+z]);
                    }
                    ///cout<<normaltMasikArfolyam.size()<<endl;
                    float szum = 0;
                    for (int k=0; k<normaltMasikArfolyam.size(); k++){
                        szum+=fabs(normaltArfolyam[k]-normaltMasikArfolyam[k]);
                    }
                    maxSzum = max(maxSzum,szum);
                }
                if (i==j) hasonlosagok[i].push_back(42);
                else hasonlosagok[i].push_back(maxSzum);
            }
        }
        ofstream ofileHas("hasonlok_102.txt");
        for (int i=0;i<hasonlosagok.size(); i++) ofileHas<<reszvenyekGPU[i].nev<<" ";
        ofileHas<<endl;
        for (int i=0;i<hasonlosagok.size(); i++){
            ofileHas<<reszvenyekGPU[i].nev<<" ";
            for (int j=0;j<hasonlosagok[i].size(); j++){
                ofileHas<<hasonlosagok[i][j]<<" ";
            }
            ofileHas<<endl;
        }
        ofileHas.close();

        for (int i=0; i<reszvenyekGPU.size(); i++){
            vector<float> minimumok(3,0);
            vector<float> minIdx(3,-1);
            for (int j=0; j<reszvenyekGPU.size(); j++){
                if (hasonlosagok[i][j]>minimumok[2]){
                    minimumok[0]=minimumok[1];
                    minimumok[1]=minimumok[2];
                    minimumok[2]=hasonlosagok[i][j];
                    minIdx[0]=minIdx[1];
                    minIdx[1]=minIdx[2];
                    minIdx[2]=j;
                } else if (hasonlosagok[i][j]>minimumok[1]){
                    minimumok[0]=minimumok[1];
                    minimumok[1]=hasonlosagok[i][j];
                    minIdx[0]=minIdx[1];
                    minIdx[1]=j;
                } else if (hasonlosagok[i][j]>minimumok[0]){
                    minimumok[0]=hasonlosagok[i][j];
                    minIdx[0]=j;
                }
            }
            float a = reszvenyekGPU[minIdx[0]].mozgoatlagokZaras[iend-vizsgaltIdo]-reszvenyekGPU[minIdx[0]].mozgoatlagokZaras[iend-vizsgaltIdo-eltolas];
            float b = reszvenyekGPU[minIdx[1]].mozgoatlagokZaras[iend-vizsgaltIdo]-reszvenyekGPU[minIdx[1]].mozgoatlagokZaras[iend-vizsgaltIdo-eltolas];
            float c = reszvenyekGPU[minIdx[2]].mozgoatlagokZaras[iend-vizsgaltIdo]-reszvenyekGPU[minIdx[2]].mozgoatlagokZaras[iend-vizsgaltIdo-eltolas];
            float hm[3];
            //if (a<0) hm[0] = reszvenyekGPU[i].mozgoatlagokZaras[iend-vizsgaltIdo+eltolas]/reszvenyekGPU[i].mozgoatlagokZaras[iend-vizsgaltIdo];
            //else
            hm[0] = reszvenyekGPU[i].mozgoatlagokZaras[iend-vizsgaltIdo+eltolas]/reszvenyekGPU[i].mozgoatlagokZaras[iend-vizsgaltIdo];
            if (a<0) hm[0]=1.0f/hm[0];
            hm[1] = reszvenyekGPU[i].mozgoatlagokZaras[iend-vizsgaltIdo+eltolas]/reszvenyekGPU[i].mozgoatlagokZaras[iend-vizsgaltIdo];
            if (b<0) hm[1]=1.0f/hm[1];
            hm[2] = reszvenyekGPU[i].mozgoatlagokZaras[iend-vizsgaltIdo+eltolas]/reszvenyekGPU[i].mozgoatlagokZaras[iend-vizsgaltIdo];
            if (c<0) hm[2]=1.0f/hm[2];
            cout<<reszvenyekGPU[i].nev<<" ";
            for (int k=0; k<3;k++) cout<<reszvenyekGPU[minIdx[k]].nev<<" "<<minimumok[k]<<" "<<hm[k]<<" ";
            cout<<endl;
        }


        return 0;
    }

    ///return 0;

    bool speedTest = false;
    if (speedTest){
        Parameterek params; params.m1=13; params.m2=19; params.m3=49;
        params.adasVeteliNapok=1;
        params.tores=0.0038f;
        params.ms=0;
        params.mi=0;
        params.buy=true;
        params.toresAlatt=false;
        //parameterek.push_back(params);

        Score score;
        vector<Score> allScores(4800*4);
        clock_t time1 = clock();
        getScore2(reszvenyekGPU,params,allScores,osszesDatum,-2);
        for (int i=0; false &&i<allScores.size();i++){
            if (allScores[i].egyNapMaximum!=0 && allScores[i].param.tores>0.0037f
                 && allScores[i].param.tores<0.0039f){

            //if (allScores[i].param==params){
                cout<<"ALMA "<<i<<" "<<allScores[i].param.m1<<" ";
                cout<<allScores[i].param.m2<<" "<<allScores[i].param.m3<<" ";
                cout<<allScores[i].param.tores<<" "<<allScores[i].param.adasVeteliNapok;
                cout<<allScores[i].param.buy<<" "<<allScores[i].param.mi;
                cout<<allScores[i].param.ms<<" "<<allScores[i].egyNapMaximum<<endl;
                ///cout<<allScores[i].teljes.size()<<endl;
                cout<<allScores[i].evVegiek[24]<<endl;
            }
        }

        //getScore(reszvenyek,params,score,osszesDatum,-2);
        //getScore(reszvenyek,params,score,osszesDatum,-2);
        //getScore(reszvenyek,params,score,osszesDatum,-2);
        getScore(reszvenyek,params,score,osszesDatum,-2);
        cout<<clock()-time1<<" "<<clock()-fullT<<endl;
        cout<<score.evVegiek[24]<<endl;
        cout<<score.egyNapMaximum<<endl;
        int hz=0;
        //for (int i=0; i<25; i++)
          cout<<  score.osszesPelda.size()<<endl;
        exit(1);
        return 0;
    }

    clock_t t = clock();
    int cnt = 0;
    vector<Parameterek> parameterek;
    for (int i1=4; i1<=21; i1++){
        for (int i2=i1+5; i2<49; i2++){
            for (int i3=i2+5; i3<=49; i3+=2){

                ///continue;
                for (int j=0; j<2; j++){
                    for (int k=0; k<6;k++){
                        for (int l=0; l<4;l++){
                            for (int b=0;b<2; b++){
                                for (int u=0; u<2; u++){
                                    for (float f=-0.0077f; f<0.0077f; f+=0.0002f){
                                        Parameterek params; params.m1=i1; params.m2=i2; params.m3=i3;
                                        params.adasVeteliNapok=j;
                                        params.tores=f;
                                        params.ms=k;
                                        params.mi=l;
                                        params.toresAlatt=(u==0);
                                        params.buy=(b==0);
                                        parameterek.push_back(params);
                                        cnt++;
                                        ///getScore(reszvenyek,params);
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }
    }

    int cnt2 = 0;
    vector<Parameterek> parameterekMA;
    for (int i1=3; i1<=21; i1++){
        for (int i2=i1+1; i2<49; i2++){
            for (int i3=i2+1; i3<=49; i3+=2){
                Parameterek params; params.m1=i1; params.m2=i2; params.m3=i3;
                parameterekMA.push_back(params);
                cnt2++;
            }
        }
    }

    //cout<<parameterekMA.size()<<endl;
    //return 0;



    bool test = false;
    if (test){
        ifstream testf("test.txt");
        ofstream oftest("oftest.txt");
        set<Pelda> tesztPeldak;
        int vizsgaltAdasvetel = 0;

        int maxEgynap = 0;

        while(!testf.eof()){
            int idx;
            testf>>idx;
            if (parameterek[idx].adasVeteliNapok!=vizsgaltAdasvetel) continue;
            Score score; getScore(reszvenyek,parameterek[idx],score,osszesDatum,-1);
            if (score.evVegiek[0]<0.90f) continue;
            //if (sco)
            oftest<<score.atlagosNapiProfit<<" "<<score.atlagosProfit<<" "<<score.egyNapMaximum<<" ";//<<endl;
            for (int i=0;i<25;i++){
                oftest<<score.alkalmakEvente[i]<<" ";
            }
            float minmax1 = 10;
            for (int i=0;i<25;i++){
                oftest<<score.evVegiek[i]<<" ";
                if (i==0){
                    minmax1=score.evVegiek[i]/100.0f;
                } else if (i<24)
                    minmax1= min(score.evVegiek[i]/score.evVegiek[i-1],minmax1);
                else {
                    minmax1= min(score.teljes[6225]/score.evVegiek[i-1],minmax1);
                }
            }
            oftest<<endl;
            if (minmax1<0.98f) continue;
            cout<<minmax1<<endl;
            tesztPeldak.insert(score.osszesPelda.begin(),score.osszesPelda.end());
        }
        oftest.close();

        ofstream oftestp("oftestPeldak.txt");
        for (Pelda pelda: tesztPeldak){
            cout<<pelda.datum.toString()<<endl;
            oftestp<<pelda.datum.toString()<<" "<<pelda.stockName<<endl;
        }

        vector<float> napiErtek; napiErtek.resize(osszesDatum.size(),100.0f);
        for (int i=0; i<osszesDatum.size(); i++){

        }
        int ev = osszesDatum[0].ev;
        for (int i=0; i<osszesDatum.size(); i++){
            ///continue;
            if (i+1<napiErtek.size())
                napiErtek[i+1]=napiErtek[i];
            if (osszesDatum[i].ev>ev){
                ev=osszesDatum[i].ev;
                ///napiErtek[i+0]=100;
                ///napiErtek[i+1]=100;
            }
            vector<float> ertekek;
            //if (datumok.find(osszesDatum[i])==datumok.end()){ continue;}
            //for (int j=0; j<osszesPelda.size(); j++){
            for (Pelda pelda: tesztPeldak){
                if (osszesDatum[i]<pelda.datum) continue;
                if (pelda.datum==osszesDatum[i]){
                    int stockIdx = pelda.reszvenyIdx;
                    int mozgoAtlagIdx = pelda.mozgoatlagIdx;
                    int idx = -1;
                    for (int j=0; j<reszvenyek[stockIdx].napok.size(); j++){
                        if (reszvenyek[stockIdx].napok[j].datum==pelda.datum){
                            idx=j;
                            break;
                        }
                    }
                    //bool felfele = reszvenyek[stockIdx].napok[idx].zaras>reszvenyek[stockIdx].mozgoatlag[params.m2].atlag[mozgoAtlagIdx-1];
                    bool felfele = true;
                    int aznapiVetel = vizsgaltAdasvetel; /// MELYIK METÓDUS
                    if (aznapiVetel==0){
                        float aznapiZaras = reszvenyek[stockIdx].napok[idx].zaras;
                        float masnapiZaras = reszvenyek[stockIdx].napok[idx+1].zaras;
                        if (felfele) ertekek.push_back(masnapiZaras/aznapiZaras-1.0f);
                        else ertekek.push_back(-(masnapiZaras/aznapiZaras-1.0f));
                    } else if (aznapiVetel==1){
                        float masnapiZaras = reszvenyek[stockIdx].napok[idx+1].zaras;
                        float harmadnapiZaras = reszvenyek[stockIdx].napok[idx+2].zaras;
                        if (felfele) ertekek.push_back(harmadnapiZaras/masnapiZaras-1.0f);
                        else ertekek.push_back(-(harmadnapiZaras/masnapiZaras-1.0f));
                    }
                }
            }
            int oszto = ertekek.size();
            float foszto = oszto;

            if (oszto>maxEgynap) maxEgynap=oszto;

            for (int j=0; j<oszto; j++){
                if (i+1<napiErtek.size()){
                    napiErtek[i+1]+=napiErtek[i]*ertekek[j]/foszto;
                }
            }
        }

        ofstream o2file("napiAlakulas0_98_24-11.txt");
        for (int i=0; i<osszesDatum.size(); i++){
            o2file<<osszesDatum[i].toStringDate()<<" "<<napiErtek[i]<<endl;
        }
        o2file.close();

        cout<<maxEgynap<<endl;

        return 0;
    }

    int thCnt = 4;
    vector<thread> szalak; szalak.resize(thCnt);
    vector<Score> scores; scores.resize(thCnt);

    vector<Score> scoresAll;

    vector<Parameterek> ptemps(parameterekMA.size()-4400);
    for (int i=4400; i<parameterekMA.size(); i++)
        ptemps[i-4400]=parameterekMA[i];
    parameterekMA = ptemps;
    /**
    vector<Parameterek> ptemps(2200);
    for (int i=2200; i<4400; i++)
        ptemps[i-2200]=parameterekMA[i];
    parameterekMA = ptemps;
    */

    list<Score> ertekek; ertekek.resize(thCnt);
    cout<<"CNT: "<<cnt<<" "<<parameterekMA.size()<<endl;
    clock_t t2 = clock();
    clock_t t3 = clock();
    ofstream ofile("nalassuk0000.txt");
    vector<vector<Score>> allScores(thCnt,vector<Score>(4800*4));
    vector<Score> saved;
    float maxProfNap = 0.0f, maxProfAtl = 0.0f, maxProf = 0.0f, minLoss = 2.0f, maxVeg = 0.0f, minVeg = 10000.0f;
    for (size_t i=0; i<parameterekMA.size();){
        int savedI = i;
        for (int j=0; j<thCnt; j++){

            //szalak[j] = thread(getScore,ref(reszvenyek),ref(parameterek[i]),ref(scores[j]),ref(osszesDatum),j);
            szalak[j] = thread(getScore2,ref(reszvenyekGPU),ref(parameterekMA[i]),ref(allScores[j]),ref(osszesDatum),j);


            //if (j<1)szalak[j] = thread(getScore2,ref(reszvenyekGPU),ref(parameterekMA[i]),ref(allScores1),ref(osszesDatum),j);
            //else szalak[j] = thread(getScore2,ref(reszvenyekGPU),ref(parameterekMA[i]),ref(allScores2),ref(osszesDatum),j);
            i++;

            //cout<<i<<endl;
            if (i>=parameterekMA.size()) break;
            ///cout<<i<<endl;
        }

        float ts = 0;
        for (int j=0; j<thCnt; j++){
            if (savedI+j>=parameterekMA.size()) continue;
            clock_t t1 = clock();
            if (szalak[j].joinable())
                szalak[j].join();
            //if ((savedI+j)%1000==0)
            if (clock()-t2>=5000){
                cout<<savedI+j<<", saved: "<<saved.size()<<", "<<clock()-t2<<" i "<<ts<<" - "<<(float)(clock()-t3)/(float)(savedI+j-0)<<endl;
                t2=clock();
            }
            //cout<<"KOK"<<endl;
            //t2=clock();
            ///cout<<savedI+j<<endl;
            ///cout<<scores[j].diffDays<<endl;
            //scoresAll.push_back(scores[j]);
            //ts+=scores[j].atlagosProfit;


            /// EZ KELL!!!
            for (int i=0; i<allScores[j].size(); i++){
                if (allScores[j][i].teljes.size()!=6225) continue;

                Score score = allScores[j][i];
                if (score.evVegiek[24]>10000 || score.evVegiek[24]<1){
                    bool toSave = false;
                    if (maxProfNap<score.atlagosNapiProfit){
                        cout<<"NAP"<<endl;
                        cout<<saved.size()<<" "; score.print();
                        maxProfNap=score.atlagosNapiProfit;
                        toSave = true;
                    }
                    if (maxProfAtl<score.atlagosProfit){
                        cout<<"ATL"<<endl;
                        cout<<saved.size()<<" "; score.print();
                        maxProfAtl=score.atlagosProfit;
                        toSave = true;
                    }
                    if (maxProf<score.maxLoss){
                        cout<<"PROF"<<endl;
                        cout<<saved.size()<<" "; score.print();
                        maxProf=score.maxLoss;
                        toSave = true;
                    }
                    if (minLoss>score.minProfit){
                        cout<<"LOSS"<<endl;
                        cout<<saved.size()<<" "; score.print();
                        minLoss=score.minProfit;
                        toSave = true;
                    }
                    if (minVeg>score.evVegiek[24]){
                        cout<<"MIN"<<endl;
                        cout<<saved.size()<<" "; score.print();
                        minVeg=score.evVegiek[24];
                        toSave = true;
                    }
                    if (maxVeg<score.evVegiek[24]){
                        cout<<"MAX"<<endl;
                        cout<<saved.size()<<" "; score.print();
                        maxVeg=score.evVegiek[24];
                        toSave = true;
                    }
                    if (toSave){
                        saved.push_back(score);
                    }
                }



                continue;
                ///Score.param. allScores[j][i].param
                /*
                ofile<<allScores[j][i].param.m1<<" "<<allScores[j][i].param.m2<<" "<<allScores[j][i].param.m3<<" "<<allScores[j][i].param.adasVeteliNapok<<" ";
                ofile<<allScores[j][i].param.buy<<" "<<allScores[j][i].param.toresAlatt<<" "<<allScores[j][i].param.mi<<" "<<allScores[j][i].param.ms<<" ";
                ofile<<allScores[j][i].param.tores<<" ";
                ofile<<allScores[j][i].atlagosNapiProfit<<" "<<allScores[j][i].atlagosProfit<<" "<<allScores[j][i].egyNapMaximum<<" ";//<<endl;
                for (int k=0;k<25;k++){
                    ofile<<allScores[j][i].alkalmakEvente[k]<<" ";
                }
                for (int k=0;k<25;k++){
                    ofile<<allScores[j][i].evVegiek[k]<<" ";
                }*/
                //for (int k=0; k<allScores[j][i].teljes.size(); k++){
                  //  ofile<<allScores[j][i].teljes[k]<<" ";
                //}
                ofile.write(reinterpret_cast<const char*>(&allScores[j][i].param.m1), sizeof(int));
                ofile.write(reinterpret_cast<const char*>(&allScores[j][i].param.m2), sizeof(int));
                ofile.write(reinterpret_cast<const char*>(&allScores[j][i].param.m3), sizeof(int));
                ofile.write(reinterpret_cast<const char*>(&allScores[j][i].param.adasVeteliNapok), sizeof(int));
                ofile.write(reinterpret_cast<const char*>(&allScores[j][i].param.buy), sizeof(bool));
                ofile.write(reinterpret_cast<const char*>(&allScores[j][i].param.toresAlatt), sizeof(bool));
                ofile.write(reinterpret_cast<const char*>(&allScores[j][i].param.mi), sizeof(int));
                ofile.write(reinterpret_cast<const char*>(&allScores[j][i].param.ms), sizeof(int));
                ofile.write(reinterpret_cast<const char*>(&allScores[j][i].atlagosNapiProfit), sizeof(float));
                ofile.write(reinterpret_cast<const char*>(&allScores[j][i].atlagosProfit), sizeof(float));
                ofile.write(reinterpret_cast<const char*>(&allScores[j][i].egyNapMaximum), sizeof(int));
                ofile.write(reinterpret_cast<const char*>(allScores[j][i].alkalmakEvente.data()), allScores[j][i].alkalmakEvente.size() * sizeof(int));
                ofile.write(reinterpret_cast<const char*>(allScores[j][i].evVegiek.data()), allScores[j][i].evVegiek.size() * sizeof(float));
                ofile.write(reinterpret_cast<const char*>(allScores[j][i].teljes.data()), allScores[j][i].teljes.size() * sizeof(float));
                //if (allScores[j][i].teljes.size()!=0){
                //cout<<allScores[j][i].teljes.size()<<" "<< allScores[j][i].evVegiek.size()<<" "<<allScores[j][i].alkalmakEvente.size()<<endl;
                //ofile.close();
                return 0;
                //}
                //ofile<<endl;
            }

        }

        ///cout<<endl<<endl<<"FOR"<<endl<<endl;

        ///Sleep(1);

    }
    ///system("cls");

    ofstream of("6600VeT2.txt");
    for (int i=0; i<saved.size(); i++){
        saved[i].op(of);
        saved[i].opt(of);
    }

    int input;
    cin>>input;
    cout<<input*2;

    /*Parameterek params; params.m1=13; params.m2=19; params.m3=49;
                            params.adasVeteliNapok=0;
                            params.tores=0.0039f;
                            cnt++;*/
                            //getScore(reszvenyek,params);
    cout<<"TIME: "<<(clock()-t)<<endl;
    //Sleep(1000);
    cout<<"TIME: "<<(clock()-t)<<endl;
    cout<<"CNT: "<<cnt<<" "<<parameterekMA.size()<<endl;

    ofile.close();




    return 0;
}
