
#include <vector>
#include <list>
#include <opencv/cv.h>
#include <opencv/ml.h>
#include <opencv/highgui.h>
#include <iostream>

#include "MSER.h"

using namespace cv;
using namespace std;


CvBoost boost;
Mat tree;

void drawMSER(cv::Mat &image, CvSeq *seq, CvSeq *contours, vector<vector<int> > &chains, vector<vector<float> > &probabilities, int chain_id, bool start_new_chain, int id, int level = 0) {


   cv::Vec3b color_vec;
   int color_val = 50 + level * 205 / 5;
   if (((CvContour *)seq)->color >= 0)
      color_vec = cv::Vec3b(0,0, color_val);
   else
      color_vec = cv::Vec3b(color_val,0,0);

   int min_y = 10000, min_x = 10000, max_x = 0, max_y = 0;

   for (int j = 0; j < seq->total; ++j) {
      CvPoint *pos = CV_GET_SEQ_ELEM(CvPoint, seq, j);
      image.at<cv::Vec3b>(pos->y, pos->x) = color_vec;
      max_y = max(max_y,pos->y);
      max_x = max(max_x,pos->x);
      min_y = min(min_y,pos->y);
      min_x = min(min_x,pos->x);
   }

   IplImage* src = cvCreateImage(cvSize(max_x-min_x+20,max_y-min_y+20),IPL_DEPTH_8U,1);
   cvZero(src);
   uchar* rsptr = (uchar*)src->imageData;
   Mat region = Mat::zeros(max_y-min_y+20,max_x-min_x+20,CV_8UC3);
   region = Scalar(255,255,255);


	for (int j = 0; j < seq->total; ++j) { 
      		CvPoint *pos = CV_GET_SEQ_ELEM(CvPoint, seq, j);
		rsptr[(10+pos->x-min_x)+(10+pos->y-min_y)*src->widthStep] = 255;
      		region.at<cv::Vec3b>((10+pos->y-min_y), (10+pos->x-min_x)) = cv::Vec3b(0,0,0);
	}
   

	vector<float> sample;
	sample.push_back(0);



	//Extract features

	int area = 0;
	int perimeter = 0;
	int num_holes = 0;
	int holes_area = 0;

	IplImage *img = cvCreateImage(cvSize(64,64),src->depth,src->nChannels);
	cvResize(src, img);

	IplImage *bw=cvCreateImage(cvSize(img->width,img->height),IPL_DEPTH_8U,1);
	IplImage *bw2=cvCreateImage(cvSize(img->width,img->height),IPL_DEPTH_8U,1);
	cvCopy(img,bw);
	cvThreshold(bw,bw2,128,255,CV_THRESH_BINARY);

	//imwrite("region-bw.jpg",Mat(bw));
	//imwrite("region-bw2.jpg",Mat(bw2));
	
	IplImage *dt=cvCreateImage(cvSize(img->width,img->height),IPL_DEPTH_8U,1);

	cvDistTransform(bw,dt,CV_DIST_L1,3,NULL); //L1 gives distance in round integers while L2 floats, in order to use L2 st should be IPL_DEPTH_32F

	uchar* dtptr = (uchar*)dt->imageData;

	double minVal;
	double maxVal;
	minMaxLoc((Mat)dt, &minVal, &maxVal);//, , Point* maxLoc=0, const Mat& mask=Mat())¶

	Scalar mean,std;
        meanStdDev(Mat(dt),mean,std,Mat(bw));


	//fprintf(stdout,"0, %f, %f, %f, ", mean[0],std[0],std[0]/mean[0]);
	sample.push_back(mean[0]);
	sample.push_back(std[0]);
	sample.push_back(std[0]/mean[0]);


	//Before releasing the image let's try to identify the number of holes in the region
	vector<vector<Point> > contours0;
	vector<Vec4i> hierarchy;
	Mat img2(bw2);
	
	findContours( img2, contours0, hierarchy, RETR_TREE, CHAIN_APPROX_SIMPLE);
	int root_contour = 0; //due to compression artifacts the contour in the zero indes could not be the main contour we are analizing
    	for (int k=0; k<hierarchy.size();k++)
    	{
	    if (contourArea(contours0.at(k))>contourArea(contours0.at(root_contour)))
		root_contour = k;
    	}
    	for (int k=0; k<hierarchy.size();k++)
    	{
	    //if ((hierarchy[k][3]==root_contour)&&((((float)contourArea(contours0.at(k))/contourArea(contours0.at(root_contour)))>0.01)||(contourArea(contours0.at(k))>31)))
	    if (hierarchy[k][3]==root_contour)
	    {
		num_holes++;
		holes_area += (int)contourArea(Mat(contours0.at(k)));
            }
    	}

	area = (int)contourArea(Mat(contours0.at(root_contour)));
	perimeter = (int)contours0.at(root_contour).size();

	//fprintf(stdout,"%d, %d, %f, %d \n", area, perimeter, (float)perimeter/area, num_holes);
	sample.push_back(area);
	sample.push_back(perimeter);
	sample.push_back((float)perimeter/area);
	sample.push_back(num_holes);
	sample.push_back((float)holes_area/area);



	cvReleaseImage(&bw);
	cvReleaseImage(&bw2);
	cvReleaseImage(&dt);
	cvReleaseImage(&src);
	cvReleaseImage(&img);






   float prediction = boost.predict( Mat(sample), Mat(), Range::all(), false, false );
   float votes      = boost.predict( Mat(sample), Mat(), Range::all(), false, true );





   int seq_index = -1;
   for (int i=id; i>=0; i--) //search for the real index of this contour
   {
        CvSeq *seq_search = *CV_GET_SEQ_ELEM(CvSeq *, contours, i);
	if (seq == seq_search)
	{
		seq_index = i;
		break;
	}
   }
   
   ostringstream s;
   s << "region-" << seq_index << ".JPG";
   imwrite(s.str(),region);
   

   // Walk all the children of this node. //TODO we can do the test
   int child = 0; 
   CvSeq *iit = seq->v_next;
   while (iit) {
      iit = iit->h_next;
      child ++;
   }
   //cout << " -------- max childs is " << child;



   if (start_new_chain) {
	 vector<int> new_chain;
         new_chain.push_back(seq_index);
	 chains.push_back(new_chain);
	 vector<float> new_prob;
         new_prob.push_back(votes);
	 probabilities.push_back(new_prob);
         chain_id=(int)chains.size()-1;
   } else {
   	chains.at(chain_id).push_back(seq_index);
	probabilities.at(chain_id).push_back(votes);
   }

   //cout << endl << "360   " << sample.at(1) << ", " << sample.at(2) << ", " << sample.at(3) << ", " << sample.at(4) << ", " << sample.at(5) << ", " << sample.at(6) << ", " << sample.at(7) << ", " << sample.at(8) << endl << endl;

   start_new_chain = false;

   if (child>1){
         start_new_chain = true;
   }


   // Walk all the children of this node. //TODO we can do the test
   bool info_shown = false; 
   CvSeq *it = seq->v_next;
   if (it == NULL) // no children -> this is a leaf
   {
      cout << "		" << seq_index << " is child of " << id  << " leaf in level " << level+1 << " with " << child << " children" << "("<<votes<<")"<< endl;
      	if (seq_index == id)
 	{
		tree.at<float>(seq_index, 0) = -1;
	} else { 
		tree.at<float>(seq_index, 0) = id; 
	}
		tree.at<float>(seq_index, 1) = tree.at<float>(id, 4);  //I'll start at this x
		tree.at<float>(seq_index, 2) = tree.at<float>(id, 5);  //I'll start at this y
		tree.at<float>(seq_index, 3) = level;
		tree.at<float>(seq_index, 4) = tree.at<float>(id, 4)+region.cols+5;
		tree.at<float>(seq_index, 5) = tree.at<float>(id, 5);
		tree.at<float>(id, 5) 	     = tree.at<float>(id, 5)+region.rows+5; //my next sibling should start at this height
		int parent_id = tree.at<float>(id, 0);
		while (parent_id >= 0)
		{
			tree.at<float>(parent_id, 5) = tree.at<float>(id, 5); //my next sibling should start at this height
			parent_id = tree.at<float>(parent_id, 0);
		}
		tree.at<float>(seq_index, 6) = region.cols;
		tree.at<float>(seq_index, 7) = region.rows;
   }
   while (it) {
      if (!info_shown)
      {
      	if (seq_index == id)
 	{
      		cout << "		" << seq_index << " is root in level with " << child << " children" << "("<<votes<<")"<< endl;
		tree.at<float>(seq_index, 0) = -1; //I'm a root
		tree.at<float>(seq_index, 1) = 5; 
		tree.at<float>(seq_index, 2) = 5;
		tree.at<float>(seq_index, 3) = level;
		tree.at<float>(seq_index, 4) = region.cols+5; //all my childrens should start at this with
		tree.at<float>(seq_index, 5) = 5; //my first children should start at this height
		tree.at<float>(seq_index, 6) = region.cols;
		tree.at<float>(seq_index, 7) = region.rows;
	} else {
      		cout << "		" << seq_index << " is child of " << id  << "-" << level << " in level " << level+1 << " with " << child << " children" << "("<<votes<<")"<< endl;
		tree.at<float>(seq_index, 0) = id; //my parent
		tree.at<float>(seq_index, 1) = tree.at<float>(id, 4);  //I'll start at this x
		tree.at<float>(seq_index, 2) = tree.at<float>(id, 5);  //I'll start at this y
		tree.at<float>(seq_index, 3) = level;
		tree.at<float>(seq_index, 4) = tree.at<float>(id, 4)+region.cols+5; //all my childrens should start at this with
		tree.at<float>(seq_index, 5) = tree.at<float>(id, 5); //my first children should start at this height
		tree.at<float>(id, 5) 	     = tree.at<float>(id, 5)+region.rows+5; //my next sibling should start at this height
		int parent_id = tree.at<float>(id, 0);
		while (parent_id >= 0)
		{
			tree.at<float>(parent_id, 5) = tree.at<float>(id, 5); //my next sibling should start at this height
			parent_id = tree.at<float>(parent_id, 0);
		}
		tree.at<float>(seq_index, 6) = region.cols;
		tree.at<float>(seq_index, 7) = region.rows;
	}
      	info_shown = true;
      }
      drawMSER(image, it, contours, chains, probabilities, chain_id, start_new_chain, seq_index, level + 1);
      it = it->h_next;
   }
}

void mserTest(const Mat &image, Mat &draw_image) {
   cv::Ptr<CvMemStorage> storage(cvCreateMemStorage(0));
   CvSeq *contours;
   //cvExtractMSER(&(IplImage)image, NULL, &contours, storage, cvMSERParams(1,10,14400,.9f,.1f)); //els params originals amb que vaig fer les primeres proves
   // amb aquest segon tenim un nombre mes alt de regions, i molta mes repetibilitat ... esta be per a veure una estructura d'arbre mes complerta (fixat sobretot en el MaX_area) pero potser es innecessari treballar amb tantes regions (la diferència es de gairebé el doble)
   // una cosa molt interessant en aquest sentit de com es representa l'arbre de CC's esta relacionada amb el criteri de maximalitat ... si vols aplicar un criteri en tot l'arbre... quin sentit te retallar el arbre en diferents branques? ... i si no, hi ha algun parametre del MSER que et permeti assegurar que els arbres que obtens tenen sentit per si sols ?
   cvExtractMSER(&(IplImage)image, NULL, &contours, storage, cvMSERParams(0,1,214400,.99f,.01f),-1); // el ultim parametre .01f provoca altissima repetibilitat (inclus pot ser zero) 
   // De cara a evitar l'alta repetibilitat en branques de l'arbre lo més lògic pot ser aplicar una regla anàloga a MinVariance després de la poda de branques unaries però únicament entre guanyadors de cada branca unaria. Així el que tens després de la poda es un arbre sense branques unaries, o sigui on cada regió o bé és una fulla o bé té 2 o més fills. 
   //En principi si el threshold es més petit que 0.5, com que els fills son non-overlapping, només hi pot haver un fill que compleixi el criteri de similitud (absDiff(parent,child)<MinDiversity) .
   //Començant pel node arrel compares cada node amb cada un dels seus fills, si n'hi ha un que compleix aquest criteri has de triar entre pare i fill escollint el que tingui més alta probabilitat de ser un caràcter. En cas que sigui el pare el que fas es eliminar el fill i connectar els nets com a fills, continuant el procés. En cas que sigui el fill, elimines el pare i aleshores tots els germans han de passar al mateix nivell que tenia el seu pare: si el pare era una arrel tots el fills passaran a ser arrels (creant nous arbres), si el pare era descendent d'un altre node, tots els germans passen a ser fills d'aquell ascendent.

   vector<vector<int> >   chains;
   vector<vector<float> > probabilities;
   bool start_new_chain = true;
   int  num_roots = 0;

   tree = Mat::zeros(contours->total,8,CV_32FC1);
   Mat winners = Mat::zeros(contours->total,1,CV_32FC1);
   vector<int> root_nodes;
   root_nodes.push_back(-1);

   for (size_t i = 0; i < contours->total; ++i) {
      CvSeq *seq = *CV_GET_SEQ_ELEM(CvSeq *, contours, i);
      // No parent, so it is a root node.
      if (seq->v_prev == NULL) {
	 root_nodes.push_back(i);
	 num_roots++;
	 cout << "Start root node " << i << endl;
         drawMSER(draw_image, seq, contours, chains, probabilities, 0, start_new_chain, i);
      }
   }

   cout << "Number of MSERs: " << contours->total << endl << endl << endl;

   for (int i = 0; i<chains.size(); i++)
   {
	int best = 0;
	float best_p = 10000;
	cout << "chain "<< i << " = ";
   	for (int j = 0; j<chains.at(i).size(); j++)
	{
		cout << ", " << chains.at(i).at(j) << "(" << probabilities.at(i).at(j) << ") ";
		if (probabilities.at(i).at(j) < best_p)
		{
			best_p = probabilities.at(i).at(j);
			best = j;
		}
	}
	cout << " -- and the winner is = " << chains.at(i).at(best);
	if (probabilities.at(i).at(best) < 0) winners.at<float>(chains.at(i).at(best),0) = 1;
	cout << endl;
   }

   cout << "Number of MSERs      : " << contours->total << endl;
   cout << "Number of roots      : " << num_roots << endl;
   cout << "Number of chains     : " << chains.size() << endl;
   cout << "Number of candidates : " << countNonZero(winners) << endl << endl << endl;

   int max_x = 0;
   int max_y = 0;
   int acc_y = 0;
   for (int j = 0; j < (root_nodes.size()-1); j++)
   {
	max_y = 0;
   	for (int i = root_nodes[j]+1; i <= root_nodes[j+1]; i++)
   	{
		max_x = max(max_x, (int)tree.at<float>(i, 1)+(int)tree.at<float>(i, 6)+10);
		max_y = max(max_y, (int)tree.at<float>(i, 2)+(int)tree.at<float>(i, 7)+10);
   	}
	acc_y = acc_y + max_y;
   }
	
   Mat composition = Mat::zeros(acc_y,max_x,CV_8UC1);

   acc_y = 0;
   max_y = 0;
   for (int j = 0; j < (root_nodes.size()-1); j++)
   {
   	for (int i = root_nodes[j]+1; i <= root_nodes[j+1]; i++)
   	{
		ostringstream s;
		s << "region-" << i << ".JPG";
		Mat region = imread(s.str(),0);
		//cout << "copying region " << i << " into " << tree.at<float>(i, 1) << " " <<  tree.at<float>(i, 2)+acc_y << " " <<  region.cols << " " << region.rows << " where the image size is " << composition.cols << " " << composition.rows << endl;
		Mat dst_roi = composition(Rect(tree.at<float>(i, 1), tree.at<float>(i, 2)+acc_y, region.cols, region.rows));
		region.copyTo(dst_roi);
		max_y = max(max_y, (int)tree.at<float>(i, 2)+(int)tree.at<float>(i, 7)+10+acc_y);
		if (winners.at<float>(i,0) == 1)
			rectangle(composition, Point(tree.at<float>(i, 1), tree.at<float>(i, 2)+acc_y), Point(tree.at<float>(i, 1)+region.cols, tree.at<float>(i, 2)+acc_y+region.rows), Scalar(128), 5, 8, 0);
	}	
	acc_y = max_y;
   }

   if (composition.rows > 45534)
   {
	Mat resized = Mat::zeros(45534,composition.cols*45534/composition.rows,CV_8UC1);
	cv::resize(composition, resized, resized.size());
     	imwrite("component_tree.JPG",resized);
   } else 
     imwrite("component_tree.JPG",composition);
		
}


int main(int argc, char *argv[]) {

    boost.load("./trained_boost.xml", "boost");

    if(argc > 1){
        Mat image = imread(argv[1], 0);
        Mat drawImage = imread(argv[1], 1);

        mserTest(image, drawImage);
        //imshow("Color mser", drawImage);
        imwrite("result.jpg", drawImage);
        //waitKey(0);
    }
}

