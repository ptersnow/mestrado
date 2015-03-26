#include "TreeNode.h"

TreeNode::TreeNode()
{
  m_name = "";
  m_aligned = NALIGNED;
  m_length = DBL_MAX;
  mp_left = NULL;
  mp_right = NULL;
}

TreeNode::~TreeNode()
{
}

void
TreeNode::init(int iNode, const char *nm)
{
  m_name = nm;
  index = iNode;
  m_aligned = ALIGNED;
}

void
TreeNode::setLength(double x)
{
  m_length = x;
}

void
TreeNode::join(int iNode, TreeNode* left, TreeNode* right, float minD, double Ti, double Tj, int nSpecies)
{
  mp_left  = left;
  mp_right = right;

  double lengthL, lengthR, half = 0.5;

  if (nSpecies == 2)
    lengthR = lengthL = (minD * half);
  else {
    lengthL = half * ( minD + (Ti - Tj) / (nSpecies - 2));
    lengthR = half * ( minD + (Tj - Ti) / (nSpecies - 2));
  }

  if (lengthL < 0.0) {
    lengthR -= lengthL;
    lengthL  = 0.0;
  }
  else if (lengthR < 0.0) {
    lengthL -= lengthR;
    lengthR  = 0.0;
  }

  index = iNode;

  mp_left->setLength(lengthL);
  mp_right->setLength(lengthR);
}

void
TreeNode::buildTreeOutput(std::ostringstream& s)
{
  std::string len;
  if (m_length == DBL_MAX)
    len = "";
  else {
    std::ostringstream t;
    t << ":" << std::setprecision(4) << std::showpoint << std::setw(8) << m_length;
    len = t.str();
  }

  if(mp_left == NULL)
    s << m_name << len;
  else {
    s << "(\n";
    mp_left->buildTreeOutput(s);
    s << ",\n";
    mp_right->buildTreeOutput(s);
    s << ")\n" << len;
  }
}

void
TreeNode::update()
{
  if(m_aligned > NALIGNED) // Update tree node to aligned
    m_aligned = ALIGNED;
}

bool
TreeNode::getNodes(std::vector<int>& nodes, int iNodes)
{
  if((mp_left->m_aligned == NALIGNED) && mp_left->getNodes(nodes, iNodes))
    return true;

  if((mp_right->m_aligned == NALIGNED) && mp_right->getNodes(nodes, iNodes))
    return true;

  if((mp_left->m_aligned > CUTTED) && (mp_right->m_aligned > CUTTED)) {
    if((nodes.size() / 3) < (iNodes - 1)) {
      nodes.push_back(index);
      nodes.push_back(mp_left->index);
      nodes.push_back(mp_right->index);

      //printf("[%d] -> [%d; %d]\n", index, mp_left->index, mp_right->index);

      m_aligned = ALIGNING;
    }
    else {
      mp_left->m_aligned = CUTTED;

      return true;
    }
  }

  return false;
}
