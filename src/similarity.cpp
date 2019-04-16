#include "similarity.hpp"



/* DOCUMENTATION
 *
 */
int overlap ()
{
    // TODO
}



/* DOCUMENTATION
 *
 */
int prefix_size (vector<string> record, int delta)
{
    return record.size() - delta + 1;
}



/* DOCUMENTATION
 *
 */
vector<string> prefix (vector<string> record, int delta)
{
    vector<string> pref;
    int prefsize = prefix_size(record,delta);
    
    for (int i = 0; i < prefsize; i++)
    {
        pref.push_back(record[i]);
    }

    return pref;
}
