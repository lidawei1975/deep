class CCommandline
{
  private:
    int narg;
    std::vector<std::string> arguments;
    std::vector<std::string> parameters;
    std::vector<std::string> informations;
    int error_flag;

    bool pharse_core(int argc, char ** argv);

  protected:
  public:
    bool pharse(int argc, char **argv);
    void init(std::vector<std::string>, std::vector<std::string>);
    void init(std::vector<std::string>, std::vector<std::string>, std::vector<std::string>);
    void print();
    std::string query(std::string);
    CCommandline();
    ~CCommandline();
};