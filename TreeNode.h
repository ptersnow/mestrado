#ifndef TREENODE_H
#define TREENODE_H

#include <cstdio>
#include <vector>
#include <string>
#include <cfloat>
#include <iomanip>
#include <sstream>

#define NALIGNED 0
#define CUTTED 1
#define ALIGNING 2
#define ALIGNED 3

class TreeNode
{
 public:
  TreeNode();
  ~TreeNode();

  void update();
  void init(int iNode, const char *nm);
  void buildTreeOutput(std::ostringstream& s);
  bool getNodes(std::vector<int>& nodes, int iNodes);
  void join(int iNode, TreeNode* left, TreeNode* right, float minD, double Ti, double Tj, int nSpecies);

 private:
  void setLength(double x);

  int index;
  short m_aligned; // 0 - not aligned; 1 - cutted; 2 - aligning; 3 - aligned
  double m_length; // Node Length
  std::string m_name;
  std::vector<int> m_nodes;

  TreeNode* mp_left;
  TreeNode* mp_right;
};

#endif /* TREENODE_H */
