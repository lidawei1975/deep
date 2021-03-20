

class cmaxclique
{
private:
    int counter;
    std::vector<int> neighbor;
    std::vector<int> nneighbor;
    std::vector< std::vector<int> > clique;
    
    int npeak;
    int pivot(std::vector<int>);
    void bk1(int c,std::vector<int> r, std::vector<int> p, std::vector<int> x);
    void bk2(int c,std::vector<int> r, std::vector<int> p, std::vector<int> x);
    
    
public:
    void init(std::vector< std::vector< int> > *n);
    void solver();
    void add_orphan();

    std::vector< std::vector<int> >  output();
    cmaxclique();
    ~cmaxclique();
};