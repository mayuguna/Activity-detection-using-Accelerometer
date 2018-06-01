package servlets;


import java.io.BufferedReader;
import java.io.BufferedWriter;
import java.io.File;
import java.io.FileWriter;
import java.io.IOException;
import java.io.InputStreamReader;
import java.net.HttpURLConnection;
import java.net.URL;
import java.sql.Connection;
import java.sql.PreparedStatement;
import java.sql.ResultSet;
import java.util.ArrayList;
import java.util.Enumeration;
import java.util.List;

import javax.servlet.ServletException;
import javax.servlet.annotation.WebServlet;
import javax.servlet.http.HttpServlet;
import javax.servlet.http.HttpServletRequest;
import javax.servlet.http.HttpServletResponse;
import javax.servlet.http.HttpSession;

import com.google.gson.Gson;
import com.google.gson.reflect.TypeToken;
import com.mysql.jdbc.Statement;

import configs.Utils;
import dbconnection.DBConnectionHandler;
import models.Row;
import models.User;

@WebServlet("/Ml")

public class MachineLearning extends HttpServlet {
	private static final long serialVersionUID = 1L;
	private Gson gson;
	private String filename;
	private ArrayList<Row> rowDrinking;
	private ArrayList<Row> rowNotDrinking;
	private int counter;
	private int counterND;

	@Override
	protected void service(HttpServletRequest request, HttpServletResponse response)
			throws ServletException, IOException {
		// TODO Auto-generated method stub
		super.service(request, response);
	}

	@Override
	protected void doGet(HttpServletRequest request, HttpServletResponse response)
			throws ServletException, IOException {
		// TODO Auto-generated method stub

		HttpSession session = null;
		System.out.println("----------------------------------Machine Servlet--------------------------------------");
		
	
	 	//String action = (String) request.getAttribute("action");

		Gson gson = new Gson();
		String json = "";

		//Enumeration paramNames = request.getParameterNames();
		//String params[] = new String[10];
		//int i = 0;
int fl=0;
			String sql3 = "select * from healthbot.sensor_records ";
			Connection con3 = DBConnectionHandler.getConnection();
			String result3 = "";

			try {
				System.out.println("i woke up :)");

				
				//PreparedStatement ps3 = con3.prepareStatement(sql3);
				/*if (ps3 == null) {
					System.out.println("Hello Error!!!");
				}
				if (params[0] != null) {
					// write a file
					
				}*/
				//String filename = 
	             File file = new File("/Applications/DailyTracker/WebContent/predict.csv");
				//String tablename = "abc";
				java.sql.Statement st=con3.createStatement();
				 sql3="SELECT id,time,x_axis,y_axis,y_axis FROM healthbot.sensor_records";
				ResultSet rs=st.executeQuery(sql3);
				
				System.out.println("about to query :)");

				//PreparedStatement ps3 = con3.prepareStatement(sql3);

				FileWriter fstream = new FileWriter(file);
	            BufferedWriter out = new BufferedWriter(fstream);
				StringBuffer content = new StringBuffer();

				String prediction="";	
				while (rs.next()) {
					
					String id=rs.getString(1);
					String time=rs.getString(2);
					String x_axis=rs.getString(3);
					String y_axis=rs.getString(4);
					String z_axis=rs.getString(5);
					String data=""+id+","+time+","+x_axis+","+y_axis+","+z_axis ;
					//if (time!=null && x_axis!=null && y_axis!=null && z_axis!=null) {
						out.write(data);
					//	System.out.println(data);

					  out.newLine();
		            fl++;}
				//}
				
				
			//	if (fl>0) {
					URL url = new URL("http://172.20.10.9:5000/predict");

					HttpURLConnection con = (HttpURLConnection) url.openConnection();

					con.setRequestMethod("GET");
				//	con.setConnectTimeout(5000);
				//	con.setReadTimeout(5000);
					int status = con.getResponseCode();
					BufferedReader in = new BufferedReader(new InputStreamReader(con.getInputStream()));
							String inputLine;
							while ((inputLine = in.readLine()) != null) {
								content.append(inputLine);
								
														}
							in.close();
							con.disconnect();
				//}
				System.out.println("posted");

				prediction=content.toString();
				 out.close();             

				 System.out.println(prediction);
				
				String[] results = { prediction, "0" };
				//System.out.println(res);
				String result = gson.toJson(results);
				response.setContentType("application/json");
				response.setCharacterEncoding("UTF-8");
				response.getWriter().write(result);
			//	ResultSet dump=st.executeQuery("truncate table healthbot.sensor_records");
				con3.close();
				
			} catch (Exception e) {
				e.printStackTrace();
			}

		

	//	System.out.println(json);

	}

	@Override
	protected void doPost(HttpServletRequest request, HttpServletResponse response)
			throws ServletException, IOException {
		doGet(request, response);
	}


}
