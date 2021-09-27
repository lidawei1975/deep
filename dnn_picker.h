//define activation funciton

enum activation_function{linear, relo, softmax};


//max pooling layer
class pool1d
{
private:
    int nfilter,npool;

public:    
    pool1d();
    ~pool1d();
    bool predict(int,std::vector<float> &, std::vector<float> &);

    inline void set_size(int m,int n) 
    {
        nfilter=m;
        npool=n;
    }
};

//base class for both dense and conv
class base1d
{
protected:
    int nfilter;
    int ninput;

    std::vector<float> kernel;
    std::vector<float> bias;
    bool mat_mul(std::vector<float> &, std::vector<float> &, std::vector<float> &, int,int,int);

public:
    bool print();
    base1d();
    ~base1d();
};

//dense connected layer
class dense: public base1d
{
private:
    enum activation_function a;
public:
    dense();
    ~dense();

    bool predict(int, std::vector<float> &, std::vector<float> &);
    bool read(std::string fname);
    int read(float *);
    inline void set_size(int n, int k) 
    {
        ninput=n;
        nfilter=k;
    }
    inline bool set_act(enum activation_function a_) {a=a_; return true;}
};



//convalution layer class
class conv1d: public base1d
{
private:
    int nkernel;

public:
    conv1d();
    ~conv1d();
    bool read(std::string);
    int read(float *);
    bool predict(int,std::vector<float> &, std::vector<float> &);

    inline void set_size(int m,int n, int k) 
    {
        nkernel=m;
        ninput=n;
        nfilter=k;
    }
};

class peak1d
{
private:

    float scale_factor;
    float noise_level; 

    int model_selection;

    int ndim,ndim0;
    int n_shift;
    class conv1d c0,c1,c2,c3,c4,c5,c6,c7;
    class pool1d p1;
    class dense d;

    bool move_mean(std::vector<float> &,int,int);
 

public: 

    std::vector<float> input;
    std::vector<int> min_flag;
    std::vector<float> output1,output2;    
    std::vector<int> posits,ptypes,shouls;
    std::vector<double> centes,sigmas,gammas,intens;
    std::vector<double> confidence;   //picked peak confidence level
    
    
    peak1d();
    ~peak1d();

    bool load();
    bool load_m2();
    bool load_m3(); 
    bool predict(std::vector<float>);
    bool moving_average_output(); 
    bool predict_step2();

    inline bool set_noise_level(float f){noise_level=f;return true;};

};


class peak2d
{
private:

    int model_selection;

    int flag1; //default: 0 for 2d spetrum. Or 1 for 2d plane of 3D spectra.

    float noise_level; //minimal peak intensity
    float user_scale;
    float user_scale2;

    std::vector<int> cx,cy; //coordiante of peaks, intersection of lines
    std::vector<int> p_2_column_paras,p_2_row_paras;
    std::vector<int> origin_xx,origin_xy,origin_yx,origin_yy; //titled peak, original point (which is always on the line)
    std::vector<int> p_2_line_column,p_2_line_row; //index to line #
    std::vector<double> inten,sigmax,sigmay,gammax,gammay,shoulx,shouly; //other peak parameters
    std::vector<double> confidencex,confidencey; 

    class peak1d p1;
    int xdim,ydim;
    std::vector<float> spectrum_column;
    std::vector<float> spectrum_row;

    //keep in mind r_column is saved col by col!!  i*ydim+j (i is along x, j is along y)
    //r_row is saved row by row
    std::vector<int> r_column,r_row;  //flag for column by column 1d 

    //all are 1D array, index is saved in column_line_index and row_line_index
    std::vector<double> c_column,c_row;  //center, with repsect to grid
    std::vector<double> a_column,a_row;  //inten
    std::vector<double> s_column,s_row;  //sigma
    std::vector<double> g_column,g_row;  //gamma
    std::vector<double> sh_column,sh_row;  //shoulders
    std::vector<double> conf_column,conf_row; //confidence level of peak

    
    //connnecting dots into lines
    std::vector<int> column_line_x,column_line_y,column_line_segment;
    std::vector<int> row_line_x,row_line_y,row_line_segment;
    std::vector<int> column_line_index,row_line_index;
    std::vector<int> rl_column,rl_row; //column and row line, in 2D matrix form, point to line index, see second part of predict_step2
    std::vector<int> rl_column_p,rl_row_p; //column and row line, in 2D matrix form, point to peak index, see second part of predict_step2
    
    //information from special_case2 to special_case3, method 0
    std::vector<int> peak_exclude;

public:
   
    

private:
    bool column_2_row(void);
    bool row_2_column(void);
    bool find_lines(int,int,std::vector<int>,std::vector<int> &,std::vector<int> &,std::vector<int> &,std::vector<int> &);
    //int intersect(double vx1,double vy1,double vx2,double vy2,double hx1,double hy1,double hx2,double hy2,double &x,double &y);
    bool check_special_case(const int,const double, const double,std::vector<int> &,std::vector<int> &,std::vector<int> &,std::vector<int> &,std::vector<int> &,std::vector<int> &,std::vector<int> &,std::vector<int> &,std::vector<int> &, std::vector<int> &);
    bool get_tilt_of_line(const int flag,const int x, const int y, const double w,const std::vector<int> &line_x,const std::vector<int> &line_y,const std::vector<int> &line_ndx, int &p1, int &p2, double &ratio);
    bool check_near_peak(const int,const int,const int,const int);
    bool predict_step1();
    bool predict_step2();
    bool predict_step3();
    // bool check_special_peaks_1();
    bool check_special_peaks_2();
    bool check_special_peaks_3();
    bool setup_peaks_from_p();
    std::vector<int> select_max_nonoverlap_set(std::vector<int>,std::vector<int>,int);

    //special case function
    bool interp2(std::vector<double>,std::vector<double>,std::vector<double> &);
    bool find_nearest_normal_peak(double x, double y,std::vector<int>,int);
    bool cut_one_peak(std::vector<double> target_line_x,std::vector<double>  target_line_y,int current_pos,std::vector<int> ndx_neighbors, int anchor_pos,int &,int &);
    bool cut_one_peak_v2(std::vector<int> &,std::vector<int> &,std::vector<int> &,const int,const std::vector<int>);

public:
    peak2d();
    peak2d(int);
    ~peak2d();

    bool init_ann(int);
    bool init_spectrum(int,int,float,float,float,std::vector<float>, int);

    
    bool predict();
    bool print_prediction();
    bool extract_result(std:: vector<double> &,std:: vector<double> &,std:: vector<double> &,std:: vector<double> &,std:: vector<double> &,std:: vector<double> &,std:: vector<double> &,std::vector<int> &,std::vector<double> &,std::vector<double> &);
    
};