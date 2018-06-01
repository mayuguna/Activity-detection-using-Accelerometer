import MySQLdb
def connection():
    conn = MySQLdb.connect(host="localhost", user = "root",passwd = "admin",db = "healthbot")
    c = conn.cursor()
    return c, conn