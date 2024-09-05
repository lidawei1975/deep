class CCommandline
{
  private:
    int narg;
    std::vector<std::string> arguments;
    std::vector<std::string> parameters;
    std::vector<std::string> informations;
    int error_flag;

    bool pharse_core(int argc, char ** argv);
    bool is_key(std::string); // check if the input is a key such as "-in". 

  protected:
  public:
    bool pharse(int argc, char **argv);
    bool pharse_file(int argc, char ** argv);
    void init(std::vector<std::string>, std::vector<std::string>);
    void init(std::vector<std::string>, std::vector<std::string>, std::vector<std::string>);
    void print();
    std::string query(std::string);
    CCommandline();
    ~CCommandline();
};