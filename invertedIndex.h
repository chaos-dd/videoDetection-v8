
#include<vector>
#include<string>

namespace vd
{

    typedef int DocId;
    typedef int WordId;

    struct IndexItem
    {
        DocId docId_;
        int wordInd;
    };
    class InvertedIndex
    {
    private:
        std::vector<std::vector<IndexItem>>  invertedIndex_;

    public:
        int querySingleWord();
    };
}